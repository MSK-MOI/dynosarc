import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from dynosarc.gen_logger import gen_logger, set_verbose
from dynosarc.dynamic.dynorc import DYNO

logger = gen_logger(__name__)
set_verbose(logger, verbose="INFO")


def desensitized_corr(data, topology, method='pearson', min_samples=None,
                      thresh=0.03, std_thresh=3, verbose="ERROR"):
    """ Compute correlation-based edge-weights for network.
    
    Parameters
    ----------
    data : pandas DataFrame
        Nodal features given with samples as columns and nodes as rows.
    topology : pandas DataFrame
        Graph topology given as edgelist with two columns labeled 'source' and 'target.'
        Note: - May result in zero or near-zero edgeweights
    method : {'pearson', 'spearman','kendall'}
        Method for computing correlation coefficient. (Default = 'pearson')
    min_samples : int { min_samples > 0 }
        Minimum number of samples required to remain for desensitized correlation evaluation. 
        (Default = 50% for pearson, 100% for spearman)
    thresh : float (thresh >= 0)
        Threshold for largest tolerated variance in leave-one-out correlations
        (Default = 0.03)
    std_thresh : float (std_thresh > 0)
        When computing the desensitized correlation, consider samples whose standardized expression is within std_thresh from the mean.
        (Default = 3)
    """
    set_verbose(logger, verbose)
    
    if method not in ['pearson','spearman','kendall']:
        raise ValueError("Unrecognized 'method' detected, must be one of {'pearson', 'spearman','kendall'}.")
    if np.any(data.isnull()):
        raise ValueError("Check data, missing values detected.")

    E = topology.copy()
    data_sub = data.copy()
    if min_samples is None:
        min_samples = round(data_sub.shape[1]/2) if method=='pearson' else data_sub.shape[1]

    C = data_sub.transpose().corr(method=method)
    if np.any(pd.isnull(C)):
        raise Exception("Null values found in correlation matrix.")
    keep = np.triu(np.ones(C.shape),1).astype('bool').reshape(C.size)
    corr = C.stack()[keep].rename_axis(index=['source','target']) # lower triangular correlation matrix
    corr = corr.to_frame(name='all')

    edges = [(aa[1].source, aa[1].target) if ((aa[1].source, aa[1].target) in corr.index) else (aa[1].target, aa[1].source) for aa in E.iterrows()]
    corr = corr.loc[edges].copy()        
        
    weighted_edgelist = corr['all'].copy() #!
    if min_samples < data.shape[1]:  # only perform sensitivity analysis if min_samples is less than the data sample size

        # check for zero variance
        corr_sample = []

        # compute correlation with leave-one-out for all samples
        for sample in tqdm(data_sub.columns, desc='Leave-one-out correlation computation', colour='green'):
            D = data_sub.drop(columns=sample, inplace=False)
            C_sub = D.transpose().corr(method=method)
            tmp = C_sub.stack()[keep].loc[edges]
            tmp.name = sample
            corr_sample.append(tmp)
        corr_sample = pd.concat(corr_sample, axis=1)

        cv = corr_sample.var(axis=1)
        logger.info('Computing correlation-based edge weights: larget variance in correlation with leave-one-out analysis = {}'.format(cv.max()))
        if cv.max() > thresh: # standardized:
            logger.info('Computing de-sensiized correlation values')
            corr_sample = corr_sample.loc[cv.index[cv>0.03]] #!
            corr_standardized = corr_sample.apply(lambda x: (x-corr_sample.mean(axis=1))/corr_sample.std(axis=1),axis=0)
            cnt = (corr_standardized.abs() <= std_thresh)

        for edx in tqdm(cv.index[cv>thresh], desc="Desensitized correlation", colour='blue'):
            # --- get samples for desensitized correlation ---
            if data_sub.loc[edx[0], cnt.loc[edx]].shape[0] >= min_samples:
                sub_samples = cnt.columns[cnt.loc[edx]].copy()
            else:
                logger.info(f"Insufficient samples for desensitized correlation between {edx[0]} and {edx[1]}, minimum samples remain.")
                sub_samples = corr_standardized.loc[edx].abs().nsmallest(n=min_samples,keep='all').index.copy()

            # --- update correlation value ---
            if data_sub.loc[edx, sub_samples].transpose().corr(method=method).isnull().any(axis=None):
                logger.info(f"Desensitized samples have zero variance, unfiltered correlation for ({edx[0]},{edx[1]}) used instead")
            else:
                weighted_edgelist[edx] = data_sub.loc[edx[0], sub_samples].corr(data_sub.loc[edx[1], sub_samples], method=method)

    weighted_edgelist = weighted_edgelist.to_frame(name='weight').loc[edges]
    E["weight"] = weighted_edgelist.values
    return E


def distance_weights(data, topology, **corr_kws):
    """ Return distance-based edge weights

    Parameters
    ---------
    data : pandas DataFrame
        Nodal features given with samples as columns and nodes as rows.
    topology : pandas DataFrame
        Graph topology given as edgelist with two columns labeled 'source' and 'target.'
        Note: - May result in zero or near-zero edgeweights
    corr_kws : dict
        Keyworkd arguments passed to `desensitized_corr` to compute the correlations.

    Returns
    -------
    s_weights : Pandas DataFrame
        Dataframe with 3 columns including source, target, weight of the network (where weights correspond to distance).
    """

    s_corr = desensitized_corr(data, topology, **corr_kws)
    s_corr = s_corr.set_index(['source', 'target'])
    s_weights = s_corr['weight'].apply(lambda x: 1. / np.sqrt(np.max([np.abs(x), 1e-4])))
    return s_weights.reset_index()



def run_curvature_simulation(directory, G, crit=0.75, force_recompute=False, chunksize=None, edgelist=None, **dyn_kws):
    """ Run dynamic curvature simulation

    Parameters
    ---------
    directory : str
        Path to store dynamic simulation results.
    G : networkx Graph
        The weighted graph (with edge weight (used to compute graph distance) attribute).
    crit : 0 < crit <= 1
        Critical scale.
    force_recompute : bool
        If True, the full dynamic curvature simulation is computed. Otherwise, if False, simulation is resumed from partial previous computation.
    dyn_kws : dict
        Keyword arguments to pass to dynamic curvature setup.

    Returns
    -------
    dyno : instance of DYNO
        Dynamic curvature simulation object    

    """

    if not os.path.isdir(directory):
        logger.info(f"Create directory to store dynamic simulation results: {directory}")
        os.mkdir(directory)

    dyno = DYNO(G, directory=directory, edgelist=edgelist, **dyn_kws)
    logger.info(f"Dynamic curvature simulation to be computed on {len(dyno.edgelist)} edges.")
    
    if os.path.isfile(os.path.join(dyno.directory, 'curvatures.csv')):
        dyno.load(kappas=True)
    dyno.compute_dynamic_curvatures(force_recompute=force_recompute, chunksize=chunksize)
    dyno.plot_dynamic_curvatures()
    dyno.plot_dynamic_curvature_variance()
    titcs = dyno.time_to_critical_curvature(crit=crit, update=True)
    titc = dyno.get_critical_timeindex(crit=crit, update=True)
    k_critical_avg = dyno.critical_curvature_avg(crit=crit, update=True)
    k_weighted_critical_avg = dyno.weighted_critical_curvature_avg(crit=crit, update=True)
    dyno.save(kappas=dyno.kappas, wassersteins=dyno.wasserstein_distances,
              distances=pd.DataFrame(data=dyno.geodesic_distances,
                                     index=range(len(dyno.G)), columns=range(len(dyno.G))),
              G=dyno.G.copy(),directory=dyno.directory)

    return dyno

def load_curvature_simulation(directory, crit=0.75, **dyn_kws):
    """ Run dynamic curvature simulation

    Parameters
    ---------
    directory : str
        Path to store dynamic simulation results.
    crit : 0 < crit <= 1
        Critical scale.
    dyn_kws : dict
        Keyword arguments to pass to dynamic curvature setup.

    Returns
    -------
    dyno : instance of DYNO
        Dynamic curvature simulation object    
    """
    
    if not os.path.isdir(directory):
        raise ValueError("Results not found")

    dyno = DYNO(nx.Graph([(0, 1), (1, 2)]), directory=directory, **dyn_kws)
    dyno.load_graph()
    dyno.load(kappas=True, wassersteins=True, distances=True)

    # titcs = dyno.time_to_critical_curvature(crit=crit, update=False)
    # titc = dyno.get_critical_timeindex(crit=crit, update=False)
    # k_critical_avg = dyno.critical_curvature_avg(crit=crit, update=False)
    # k_weighted_critical_avg = dyno.weighted_critical_curvature_avg(crit=crit, update=False)
    return dyno #, k_critical_avg, k_weighted_critical_avg 


def get_edges(G, label, v):
    """ Edges in graph with attribute less than specified threshold """
    return [k for k in G.edges() if G.edges[k][label] < v]


def updateCC(CC, g, w_tag, v):
    """ update CC matrix for recording shared connected components """
    edges = get_edges(g, w_tag, v)
    g.remove_edges_from(edges)
    ccn = list(nx.connected_components(g))
    for c in ccn:
        for x in c:
            for y in c:
                CC[x, y] += 1


def community_hierarchy(G, attr, method='ward', optimal_ordering=True, cc_fname=None, cut_uniform=False, mx_bump=100, thresh_cut=0.0):
    """ Create hierarchy based off shared connected compenents as edges are filtered out by edge attribute

    Parameters
    ----------
    G : NetworkX graph
        Graph object.
    attr : str
        Edge attribute to filter edges.
    method : str
        Method for heriarchical clustering.
    optimal_ordering : bool
        Re-order for improved visualization.
        If True, the linkage matrix will be reordered so that the distance
        between successive leaves is minimal (i.e., ordering of (i,j) in each merge).
    cc_fname : str
        Filename to load/store CCrecord so it doesn't need to be recomputed by layout.
            If given, if CCrecord already exists, it will be loaded instead of recomputing;
            if it don't exist, CCrecord will be computed and stored to the given filename.
    cut_uniform : bool
        If True, remove edges based on uniformly spaced threshold where the increment is either the minimum difference between
            consecutive edge weights or 1e-4, which ever is larger. If False, remove edges based on sorted edge weight values.
    
    Return
    ------
    CCrecord : pandas DataFrame
        CCrecord.iloc[i,j] = the number of times nodes i and j were in the same connected component.
    CCdist : pandas DataFrame
        CCdist.iloc[i,j] = distance between nodes i and j based off of CCrecord.
    linkage : numpy ndarray
        Linkage matrix based off CCdist.
    """
    assert method in ['single','complete','average','weighted',
                     'centroid','median','ward'], "Unrecognized method."
    assert nx.get_edge_attributes(G, attr), "Edge weight not detected."

    if cc_fname and os.path.isfile(cc_fname):
        # logger.info(f"Loading CCrecord from {cc_fname}")
        CCrecord = pd.read_csv(cc_fname, header=0, index_col=0)
    else:
        # logger.debug("Computing CCrecord")
        w = list(nx.get_edge_attributes(G, attr).values())
        w_sorted = sorted(list(set([np.round(k, 4) for k in w if k > thresh_cut])))
        CC = np.zeros((len(G), len(G)))
        g = G.copy()
        updateCC(CC, g, attr, thresh_cut)

        if cut_uniform:
            dx = max(1e-4, min([xj - xi for xi, xj in zip(w_sorted, w_sorted[1:])]))
            # w_sorted = np.linspace(thresh_cut, max(w_sorted), num=min(1000, int(max(w_sorted)/1e-4)), endpoint=True)[1:]  # don't include ~0 (or min value set as thresh_cut)
            w_sorted = np.linspace(thresh_cut, max(w_sorted), num=int(max(w_sorted)/dx), endpoint=True)[1:]  # don't include ~0 (or min value set as thresh_cut) 
    
        for ww in tqdm(w_sorted,colour='green'):
            updateCC(CC, g, attr, ww)
            
        CCrecord = pd.DataFrame(data=CC,index=[G.nodes[k]['name'] for k in range(len(G))],
                                columns=[G.nodes[k]['name'] for k in range(len(G))])

        if cc_fname:
            CCrecord.to_csv(cc_fname, header=True, index=True)
            logger.info(f"CCrecord saved to {cc_fname}")

    mx = CCrecord.max().max() # only communities that persist to `mx` are islands 
    CCdist = mx - CCrecord.copy()
    CCdist.values[CCdist.values==mx] = mx + mx_bump
    linkage = hc.linkage(sp.distance.squareform(CCdist.copy()), method=method, optimal_ordering=optimal_ordering)
    
    # silh_scores = {}
    # for cut in linkage[:-1, 2]:
    #     labels = hc.fcluster(linkage, cut, criterion='distance')
    #     silh_scores[cut] = silhouette_score(CCdist, labels, metric="precomputed")
    # silh_scores = pd.Series(silh_scores)
    # opt_cut = silh_scores.idxmax()
    # logger.info(f"Optimal Silhouette score = {np.round(silh_scores[opt_cut], 4)} with optimal cut = {np.round(opt_cut, 4)}")
    return CCrecord, CCdist, linkage # , opt_cut


def plot_dendr(linkage, labels,
               no_plot, ttl=None,
               leaf_font_size=7, color_threshold=1000, line_width=1.0,
               orientation="top", leaf_rotation=90, figsize=(17, 8)):
    """ plot dendrogram

    Parameters
    ---------- 
    linkage : linkage matrix
    labels : list
        Labels for branches in dendrogram figure
    leaf_font_size : int
        Fontsize for leaf labels in dendrogram figure.
    color_threshold : float
        Color threshold for dendrogram.
    line_width : float
        Width of dendrogram lines.
    no_plot : bool
        If True, no figure with the dendrogram plotted is shown
    ttl : str
        Title for dendrogram figure.

    Return
    ------
    nodes : scipy.cluster.hierarchy.dendrogram
        The resulting dendrogram object.
    """
    # dendrogram options
    plt.rcParams['lines.linewidth'] = line_width
    hc.set_link_color_palette(['m', 'y', 'blue', 'darkgreen', 'orange'])
    truncate_mode = None
    get_leaves = True
    count_sort = False  
    distance_sort = 'ascending'
    show_leaf_counts = False 

    # dendrogram and figure
    if no_plot:
        ax = None
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    nodes = hc.dendrogram(linkage, truncate_mode=truncate_mode, ax=ax,
                          color_threshold=color_threshold,  
                          get_leaves=get_leaves, labels=labels, leaf_font_size=leaf_font_size,
                          above_threshold_color="gray",
                          count_sort=count_sort, distance_sort=distance_sort, no_plot=no_plot,
                          show_leaf_counts=show_leaf_counts, leaf_rotation=leaf_rotation, orientation=orientation)
    if not no_plot:
        if ttl:
            _ = plt.title(ttl)
        fig.tight_layout()
        plt.show()
    plt.rcParams['lines.linewidth'] = 1.5  # return to matplotlib default line width
    hc.set_link_color_palette(None)  # reset dendrogram palette
    return nodes

