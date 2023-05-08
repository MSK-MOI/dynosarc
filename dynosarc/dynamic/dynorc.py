"""

Description:
------------

A class to compute dynamic Ollivier-Ricci curvature 
on the edge of a graph
based off of diffusion Markov processes using
nodal and edge weights in a given Networkx graph. 

Credit:
-------
Code based off of the Python geometric_clustering package written by Gosztolai and Arnaudon.
        Original code can be found at https://github.com/agosztolai/geometric_clustering.
```
@inproceedings{gosztolaiArnaudon,
  author    = {Adam Gosztolai and
               Alexis Arnaudon},
  title     = {Unfolding the multiscale structure of networks with dynamical Ollivier-Ricci curvature},
  bookTitle = {arXiv},
  year      = {2021}
}
"""
import os
import time

from itertools import chain

import cvxpy as cvx
import matplotlib.pyplot as plt
import multiprocessing as mp
import networkx as nx
import numpy as np
import ot
from functools import partial
import pandas as pd
import scipy as sc
import scipy.sparse.csgraph as scg
from tqdm import tqdm

# from warnings import simplefilter
# simplefilter(action='ignore', category=FutureWarning) # # ignore all future warnings

from dynosarc.gen_logger import gen_logger
from dynosarc.gen_logger import set_verbose 

logger = gen_logger(__name__)

#######################
# ~~~ Local methods ~~~
#######################

def _get_times(times=None, t_min=-1.5, t_max=1.0, n_t=20, log_time=True):
    """ get array of simulation time-points """
    if times is not None:
        return times
    else:        
        assert n_t > 0, "At least one time step is required for the dynamic simulation."
    if log_time:
        assert t_min < t_max, "Initial simulation time must be before the final time (t_min < t_max)."
        return np.logspace(t_min, t_max, n_t)
    else:
        assert 0 < t_min < t_max, "Linearly spaced simulation time-points must satisfy 0 < t_min < t_max."
        return np.linspace(t_min, t_max, n_t)


def _compute_distance_geodesic(G, weight="weight"):
    """Geodesic distance matrix."""

    assert nx.get_edge_attributes(G, weight), f"Edge weight '{weight}' not detected in graph, distance matrix not computed."
    gd = None
    try:
        gd = scg.dijkstra(
            nx.adjacency_matrix(G, weight=weight, nodelist=list(range(G.number_of_nodes()))),
            directed=True, unweighted=False
        )
    except:
        if gd is None:
            raise ValueError("Couldn't compute geodesic distances.")
    return gd

def _construct_laplacian(graph, weight="adj", use_spectral_gap=True):
    """ Laplacian matrix. """
    if weight:
        assert nx.get_edge_attributes(graph, weight), \
            f"Edge attribute '{weight}' not found in graph, Laplacian matrix not constructed."
    nl = list(range(graph.number_of_nodes()))
    degrees = np.array([graph.degree(i, weight=weight) for i in nl])
    laplacian = None
    try:        
        laplacian = nx.laplacian_matrix(graph, weight=weight, nodelist=nl).dot(sc.sparse.diags(1.0 / degrees))
    except:
        if laplacian is None:
            raise ValueError("Couldn't compute Laplacian")

    if use_spectral_gap and len(graph) > 3:
        spectral_gap = abs(sc.sparse.linalg.eigs(laplacian, which="SM", k=2)[0][1])
        logger.debug("Spectral gap = 10^{:.1f}".format(np.log10(spectral_gap)))
        laplacian /= spectral_gap

    return laplacian


def _heat_kernel(measure, laplacian, timestep):
    """Compute matrix exponential on a measure."""
    return sc.sparse.linalg.expm_multiply(-timestep * laplacian, measure)


def optimal_transportation_distance(x, y, d, solvr=None):
    """ Compute the optimal transportation distance (OTD) of the given density distributions by CVXPY. 
    Parameters:
    -----------
    x : (m,) numpy.ndarray  
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix. 

    Returns:
    ---------
    m : float
        Optimal transportation distance. 
    """
    logger.info("Explicit OTD computation")
    rho = cvx.Variable((len(y), len(x)))  # the transportation plan rho
    # objective function d(x,y) * rho * x, need to do element-wise multiply here
    obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))
    # \sigma_i rho_{ij}=[1,1,...,1]
    source_sum = cvx.sum(rho, axis=0, keepdims=True)
    # constrains = [rho * x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
    constrains = [rho @ x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
    prob = cvx.Problem(obj, constrains)
    if solvr is None:
        m = prob.solve(verbose=True)
    else:
        m = prob.solve(solver=solvr, verbose=True)
    logger.info("Completed explicit OTD computation")
    return m


def OTD(x, y, d, solvr=None):
    """ Compute the optimal transportation distance (OTD) of the given density distributions 
    trying first with POT package and then by CVXPY. 

    Parameters:
    -----------
    x : (m,) numpy.ndarray  
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix. 

    Returns:
    ---------
    m : float
        Optimal transportation distance. 
    """
    try:
        wasserstein_distance, lg = ot.emd2(x, y, d, log=True, numItermax=150000)
        if lg['warning'] is not None:
            logger.info(f"POT library failed: warning = {lg['warning']}, retry with explicit computation")
            wasserstein_distance = optimal_transportation_distance(x, y, d)
            # wasserstein_distance = np.nan
            # wasserstein_distance += 10000.
    except cvx.error.SolverError:
        logger.info("OTD failed, retry with SCS solver")
        wasserstein_distance = optimal_transportation_distance(x, y, d, solvr='SCS')
        # wasserstein_distance = np.nan
    return wasserstein_distance

def _edge_OTD(
        edge,
        measures,
        geodesic_distances,
        measure_cutoff=1e-6,
        sinkhorn_regularisation=0,
        balanced=True,
        **kwargs,
):
    """Compute OTD for an edge (could be a fictitious edge).

    Parameters
    ----------
    **args : extra parameters for calls to OMT solvers.

"""
    node_x, node_y = edge
    m_x, m_y = measures[node_x], measures[node_y]

    Nx = np.where(m_x >= measure_cutoff * np.max(m_x))[0]
    Ny = np.where(m_y >= measure_cutoff * np.max(m_y))[0]

    m_x, m_y = m_x[Nx], m_y[Ny]
    if balanced:
        m_x /= m_x.sum()
        m_y /= m_y.sum()

    distances_xy = geodesic_distances[np.ix_(Nx, Ny)]

    if sinkhorn_regularisation > 0:
        wasserstein_distance = ot.sinkhorn2(m_x, m_y, distances_xy, sinkhorn_regularisation)[0]
    else:
        wasserstein_distance = OTD(m_x, m_y, distances_xy)

    return wasserstein_distance


def _edge_curvature(
        edge,
        measures,
        geodesic_distances,
        measure_cutoff=1e-6,
        sinkhorn_regularisation=0,
        weighted_curvature=False,
        OTD_record=True,
        balanced=True,
        **kwargs,
):
    """Compute curvature for an edge.

    Parameters
    ----------
    OTD_record : boolean 
        Option to return Wasserstein distance along with curvature. (Default value = True).
    **kwargs : extra parameters for calls to OMT solvers.
    """
    wasserstein_distance = _edge_OTD(edge,
                                     measures,
                                     geodesic_distances,
                                     measure_cutoff=measure_cutoff,
                                     sinkhorn_regularisation=sinkhorn_regularisation,
                                     balanced=balanced,
    )

    node_x, node_y = edge

    if weighted_curvature:
        curv = geodesic_distances[node_x, node_y] - wasserstein_distance
    else:
        curv = 1.0 - wasserstein_distance / geodesic_distances[node_x, node_y]

    if OTD_record:
        return curv, wasserstein_distance
    else:
        return curv


def save_curvature(kappas=None, wassersteins=None, distances=None, G=None, directory="dynamic_curvatures"):
    """Save curvatures as csv.

    Parameters
    ----------
    kappas : pandas DataFrame
        Dataframe of curvatures with edges as columns and times as rows
    wassersteins : pandas DataFrame
        Dataframe of Wasserstein distances with edges as columns and times as rows
    distances : pandas DataFrame
        Dataframe of weighted hop distance between every two nodes
    G : networkx Graph
        Graph to be saved as .graphml file
    directory : str
        Path of directory where to save data

    Output files:
    ------------

    """

    if not os.path.isdir(directory):
        os.mkdir(directory)
        logger.info(f"Creating directory: {directory}")

    if kappas is not None:
        kappas.to_csv(os.path.join(directory,"curvatures.csv"), header=True, index=True)
        logger.debug(f"Kappas saved to {os.path.join(directory,'curvatures.csv')}")

    if wassersteins is not None:
        wassersteins.to_csv(os.path.join(directory, "wasserstein_distances.csv"),header=True, index=True)
        logger.debug(f"Wasserstein distances saved to {os.path.join(directory,'wasserstein_distances.csv')}")

    if distances is not None:
        distances.to_csv(os.path.join(directory, "graph_distances.csv"), header=True, index=True)
        logger.debug(f"Graph distances saved to {os.path.join(directory,'graph_distances.csv')}")

    if G is not None:
        g = G.copy()
        for ky, vl in G.graph.items():
            if isinstance(vl, list):
                logger.debug(f"Need to remove list value for key '{ky}' in G.graph.items() in order to save as graphml")
                vlp = g.graph.pop(ky)
                logger.debug(f"Key '{ky}' removed from G.graph.items()")

        nx.write_graphml(g, os.path.join(directory, "dyno_graph.graphml"))
        logger.debug(f"Graph saved to {os.path.join(directory,'dyno_graph.graphml')}")
    
def _compute_dynamic_curvatures(G: nx.Graph,
                                times,
                                directory="dynamic_curvatures",
                                geodesic_distances=None,
                                e_weight="weight",
                                use_spectral_gap=False,
                                edgelist=None,
                                proc=mp.cpu_count(),
                                chunksize=None,
                                measure_cutoff=1e-6,
                                sinkhorn_regularization=0,
                                weighted_curvature=False,
                                display_all_positive=False,
                                kappas=None,
                                ):
    """ Compute curvatures for all edges in edgelist (default: all edges in graph)

    Parameters
    ---------
    directory : str
        Path of directory where to save data. If results should not be saved, set to None.
    kappas : numpy array
        (Optional) A t x m array of precomputed curvatures for t-time steps and m-edges.
        If provided, dynamic curvature simulation is resumed from last provided time-step.
    """
    
    if edgelist is None:
        edgelist = list(G.edges())
            
    if geodesic_distances is None:
        logger.info("Computing geodesic distances")
        t0 = time.time()
        geodesic_distances = _compute_distance_geodesic(G, weight=e_weight)
        logger.debug(f"Geodesic distance matrix computed in {time.time() - t0} seconds")
    logger.debug("Assigning geodesic distances to edge weight attribute 'distance'")
    nx.set_edge_attributes(G, {(v0, v1): geodesic_distances[v0, v1].item() for v0, v1 in G.edges()}, name="distance")

    logger.info("Constructing Laplacian")
    dmax = float(geodesic_distances.max()) 
    nx.set_edge_attributes(G, {ee: dmax - G.edges[ee]["distance"] for ee in G.edges()}, name="adj")
    laplacian = _construct_laplacian(G, weight="adj", use_spectral_gap=use_spectral_gap)

    times_with_zero = np.insert(times, 0, 0.0)
    if kappas is None:
        kappas = np.ones([len(times), len(edgelist)])
        start_index = 0
    else:
        if kappas.shape[1] != len(edgelist):
            raise ValueError("Number of columns in preloaded curvature must match the size of the edgelist.")
        start_index = kappas.shape[0] 
        kappas = np.concatenate((kappas, np.ones([len(times) - kappas.shape[0], len(edgelist)])), axis=0)
        
    measures = list(np.eye(len(G)))
    # wasserstein_distances = np.zeros([len(times), len(edgelist)])

    logger.info("Computing all curvatures")
    with mp.Pool(proc) as pool:
        if chunksize is None:
            chunksize = max(1, int(len(edgelist) / proc))

        # for time_index in tqdm(range(len(times)), desc="Dynamic simulation", colour="green"):
        for time_index in tqdm(range(start_index, len(times)), desc="Dynamic simulation", colour="green"):
            logger.debug("---------------------------------")
            logger.debug("Step %s / %s", str(time_index), str(len(times)))
            logger.debug("Computing diffusion time 10^{:.1f}".format(np.log10(times[time_index])))

            logger.debug("Computing measures")
            if (start_index > 0) and (time_index == start_index):
                ts = times_with_zero[time_index + 1] - times_with_zero[0]
            else:
                ts = times_with_zero[time_index + 1] - times_with_zero[time_index]
                
            measures = pool.map(
                partial(
                    _heat_kernel, 
                    laplacian=laplacian,
                    timestep=ts, # times_with_zero[time_index + 1] - times_with_zero[time_index],
                ),
                measures,
                chunksize=chunksize,
            )

            logger.debug("Computing curvatures")
            kappas[time_index] = pool.map(
                partial(
                    _edge_curvature, 
                    measures=measures,
                    geodesic_distances=geodesic_distances,
                    measure_cutoff=measure_cutoff,
                    sinkhorn_regularization=sinkhorn_regularization,
                    weighted_curvature=weighted_curvature,
                    OTD_record=False,
                ),
                edgelist,  
                chunksize=chunksize,
            )

            if all(kappas[time_index] > 0) and display_all_positive:
                logger.info("All edges have positive curvatures, so you may stop the computations")
                display_all_positive = False

            kappas_tmp = pd.DataFrame(data=kappas[:time_index+1],
                                      index=times[:time_index+1],
                                      columns=pd.MultiIndex.from_arrays([[st[0] for st in edgelist],
                                                                         [st[1] for st in edgelist]],
                                                                        names=('source', 'target')))
            if directory:
                save_curvature(kappas=kappas_tmp, directory=directory) 

    return pd.DataFrame(data=kappas.copy(), index=times.copy(),
                        columns=pd.MultiIndex.from_arrays([[st[0] for st in edgelist],
                                                           [st[1] for st in edgelist]],
                                                          names=('source', 'target')))

########################
# ~~ Class definition ~~
########################


class DYNO:
    """ A class to compute dynamic Ollivier-Ricci curvature on the edges
    of a simple, undirected, connected NetworkX graph based on diffusion
    Markov processes using nodal and edge weights 
    
    Parameters
    ----------

    G : NetworkX graph
        A given (undirected, connected) NetworkX graph.
    times : {3-tuple (t_min,t_max,n_t),  array}
        Specify time steps to compute curvature of the dynamic simulation.

        input options:
        -------------
        tuple (start,stop,n_t) : 
            Provide a tuple with start and stop times and the number of time steps (n_t) for the simulation.
            The discrete time-points at which curvature is computed 
            are 'n_t' evenly spaced points on the log10 scale between 'start' and 'stop.'
       
        array :
            Array with discrete time-points at which to compute curvature.   
            
    use_spectral_gap : bool
        If True, normalize time by the spectral gap of laplacian
    e_weight : str
        Edge attribute used as weight for computing the weighted hop distance
    proc : int                                                                 
        Number of processor used for multiprocessing. (Default value = cpu_count()). 
    measure_cutoff : float
        Cutoff of the measures, in [0, 1], with no cutoff at 0    
    sinkhorn_regularization : float 
        Sinkhorn regularization, when 0, no sinkhorn is applied
    weighted_curvature : bool 
        If True, the curvature is multiplied by the edge distance
    directory : str
        Path to save curvatures at each time step and results. If results should not be saved, set to None.
    verbose ("DEBUG","INFO","WARNING","ERROR"): set verbose level, default = "INFO"
    
    edgelist : list
    geodesic_distances : numpy ndarray
    display_all_positive : bool (Default = False)
    
    Comments:
    --------
    - Any self-loops in the graph are removed
    - If the graph is not connected, it is reduced to the largest connected component for analysis
    """

    def __init__(self,
                 G: nx.Graph,
                 times=None,
                 t_min=-2.0,
                 t_max=1.5,
                 n_t=10,
                 log_time=True,
                 use_spectral_gap=True,
                 e_weight="weight",
                 proc=mp.cpu_count(),
                 measure_cutoff=1e-6,
                 sinkhorn_regularization=0,
                 weighted_curvature=False,
                 directory=None,
                 verbose="INFO",
                 edgelist=None,
                 geodesic_distances=None,
                 display_all_positive=False,
                 ):

        self.G = G.copy()
        self.use_spectral_gap = use_spectral_gap
        self.e_weight = e_weight
        self.proc = proc
        self.measure_cutoff = measure_cutoff
        self.sinkhorn_regularization = sinkhorn_regularization
        self.weighted_curvature = weighted_curvature
        self.directory = "dynamic_curvature_results" if directory is None else directory
        self.verbose = verbose
        self.edgelist = edgelist
        self.geodesic_distances = geodesic_distances
        self.display_all_positive = display_all_positive
        self.times = _get_times(times=times, t_min=t_min, t_max=t_max, n_t=n_t, log_time=log_time)
        
        self.set_verbose(verbose)        

        # --- check graph for self-loops ---
        self_loop_edges = list(nx.selfloop_edges(self.G))
        if self_loop_edges:
            logger.warning('Removing {:d} self-loop edges'.format(len(self_loop_edges)))
            self.G.remove_edges_from(self_loop_edges)

        # --- check graph is connected ---
        if not nx.is_connected(self.G):
            logger.warning("Graph is not connected, reduce to largest connected component instead")
            self.G = nx.Graph(self.G.subgraph(max(nx.connected_components(self.G), key=len)))

        # --- relabel nodes from 0 to N-1 where N = number of nodes ---
        if not nx.get_node_attributes(self.G, name='name'):
            self.G = nx.convert_node_labels_to_integers(self.G, label_attribute='name')
        self.name2node = {v: k for k, v in nx.get_node_attributes(self.G, "name").items()}

        # --- update edgelist ---
        if edgelist is None:
            self.edgelist = list(self.G.edges())
        else:
            # update edgelist to have names corresponding to integar indices
            self.edgelist = [(self.name2node[ee[0]], self.name2node[ee[1]]) for ee in edgelist]

    def set_verbose(self, verbose):
        """ Set the verbose level for this process. 

        Parameters
        ----------
        verbose : {"INFO", "TRACE","DEBUG","ERROR"}
        Verbose level. (Default value = "ERROR")
            - "INFO": show only iteration process log.
            - "TRACE": show detailed iteration process log.
            - "DEBUG": show all output logs.
            - "ERROR": only show log if error happened.
        """
        self.verbose = verbose
        set_verbose(logger, verbose)
        # logger.info(f"Set verbosity to {self.verbose}")

    def get_node_index(self, node=None):
        """ Get index of node by original label.

        Parameters
        ----------
        node : label of node in original graph
            label of node to return index.
            When no node is specified, the default behavior is to
            return the entire dictionary mapping node names to indices.

        Returns
        -------
        node_name : int
            When the node name is given, the node index is returned.
        node_names : dict
            If no node name is given, a copy of the dictionary mapping node name to index is returned for all nodes
        """
        if node is None:
            return self.name2node
        
        assert node in self.name2node.keys(), "Unrecognized node label."
        return self.name2node[node]

    def whop_distance_matrix(self, recompute=False):
        """ Compute geodesic ditances from edge weight self.G.edges[w_weight] """
        if recompute or (self.geodesic_distances is None):
            t0 = time.time()
            self.geodesic_distances = _compute_distance_geodesic(self.G, weight=self.e_weight)
            logger.info(f"Geodesic distance matrix computed in {time.time() - t0} seconds")
        return self.geodesic_distances

    def compute_dynamic_curvatures(self, force_recompute=False, chunksize=None):
        """ Compute curvatures for all edges.
        Note: if kappas from partial simulation already exist (self.kappas), then the dynamic curvature simulation is resumed from
        the last computed time-step unless force_recompute option is used."""
        
        if self.geodesic_distances is None:
            logger.info("Computing geodesic distances")
            t0 = time.time()
            self.geodesic_distances = _compute_distance_geodesic(self.G, weight=self.e_weight)
            logger.debug(f"Geodesic distance matrix computed in {time.time() - t0} seconds")
        nx.set_edge_attributes(self.G,
                               {(v0, v1): self.geodesic_distances[v0, v1] for v0, v1 in self.G.edges()},
                               name="distance")

        try:
            kappas = getattr(self, 'kappas')
            if force_recompute:
                kappas = None
                logger.info("Forced recompute detected, the full dynamic curvature simulation will be computed")
            else:
                # --- check that columns are ordered correctly ---
                if kappas.columns.tolist() != self.edgelist:
                    if set(kappas.columns) == set(self.edgelist):
                        kappas = kappas[self.edgelist]
                    else:
                        raise ValueError("Incosistent edgelist.")
                kappas = kappas.values
                logger.info("Resuming computation of dynamic curvature simulation")
        except AttributeError:
            logger.info("No prior partial simulation detected, the full dynamic curvature simulation will be computed")
            kappas = None

            
        self.kappas = _compute_dynamic_curvatures(self.G,
                                                  self.times,
                                                  directory=self.directory,
                                                  geodesic_distances=self.geodesic_distances,
                                                  e_weight=self.e_weight,
                                                  use_spectral_gap=self.use_spectral_gap,
                                                  edgelist=self.edgelist,
                                                  proc=self.proc,
                                                  chunksize=chunksize,
                                                  measure_cutoff=self.measure_cutoff,
                                                  sinkhorn_regularization=self.sinkhorn_regularization,
                                                  weighted_curvature=self.weighted_curvature,
                                                  display_all_positive=self.display_all_positive,
                                                  kappas=kappas,
                                                  )

        # --- Record Wasserstein distances ---
        geo_dists = np.asarray([self.geodesic_distances[source, target] for (source, target) in self.edgelist])
        if self.weighted_curvature:  # k = d-W => W=d-k
            wasserstein_distances = geo_dists - self.kappas
        else:  # k = 1-W/d => W=d(1-k)
            wasserstein_distances = geo_dists*(1.0-self.kappas)
        self.wasserstein_distances = wasserstein_distances.copy()

        # --- save results ---
        if self.directory:
            save_curvature(wassersteins=wasserstein_distances,
                           distances=pd.DataFrame(data=self.geodesic_distances,
                                                  index=range(len(self.G)), columns=range(len(self.G))),
                           G=self.G, directory=self.directory)



    def load(self, kappas=True, wassersteins=False, distances=False):
        """ load curvatures and data """

        # kappas
        if kappas and os.path.isfile(os.path.join(self.directory, "curvatures.csv")):
            kappas_tmp = pd.read_csv(os.path.join(self.directory, "curvatures.csv"), header=[0, 1], index_col=0)
            kappas_tmp.rename(columns={c: int(c) for c in chain(*kappas_tmp.columns)}, inplace=True)
            self.kappas = kappas_tmp
            logger.debug("Loaded kappas from file")

        # wassersteins
        if wassersteins and os.path.isfile(os.path.join(self.directory, "wasserstein_distances.csv")):
            wassersteins_tmp = pd.read_csv(os.path.join(self.directory, "wasserstein_distances.csv"), header=[0, 1], index_col=0)
            wassersteins_tmp.rename(columns={c:int(c) for c in chain(*wassersteins_tmp.columns)}, inplace=True)
            self.wasserstein_distances = wassersteins_tmp
            logger.debug("Loaded Wasserstein distances from file")

        # graph distance
        if distances and os.path.isfile(os.path.join(self.directory, "graph_distances.csv")):
            self.geodesic_distances = pd.read_csv(os.path.join(self.directory, "graph_distances.csv"), header=0, index_col=0).values
            logger.debug("Loaded graph distances from file")


    def save(self, kappas=None, wassersteins=None, distances=None, G=None, directory=None):
        """Save curvatures as csv.

        Parameters
        ----------
        kappas : pandas DataFrame
            Dataframe of curvatures with edges as columns and times as rows
        wassersteins : pandas DataFrame
            Dataframe of Wasserstein distances with edges as columns and times as rows
        distances : pandas DataFrame
            Dataframe of weighted hop distance between every two nodes
        G : networkx Graph
            Graph to be saved as .graphml file
        directory : str
            Path of directory where to save data

        Output files:
        ------------
        
        """

        if directory is None:
            directory = self.directory
        save_curvature(kappas=kappas, wassersteins=wassersteins, distances=distances, G=G, directory=directory)

            
    def save_graph(self, directory=None):  
        """ save graph """
        if directory is None:
            directory = self.directory

        self.save(G=self.G, directory=directory)


    def load_graph(self, directory=None, node_type=int):  
        """ save graph """
        if directory is None:
            directory = self.directory

        assert os.path.isfile(os.path.join(directory,
                                           "dyno_graph.graphml")), f"Graph file not found: {os.path.join(directory, 'dyno_graph.graphml')}."
        logger.debug(f"Graph loaded from {os.path.join(directory, 'dyno_graph.graphml')}")
        self.G = nx.read_graphml(os.path.join(directory, "dyno_graph.graphml"),
                                 node_type=node_type)
        self.name2node = {v: k for k, v in nx.get_node_attributes(self.G, "name").items()}


    def get_critical_timeindex(self, crit=0.75, update=False):
        """ Return index of critical time defined as the index of the first time point
        that the curvature reaches critical value (crit) 
        (NOTE: could be for a fictitious edge.)

        Parameters
        ----------
        crit : float (0.0 <= crit <= 1.0) 
            Critical curvature value.
        update : boolean
            If True, saves critical curvatures as an edge attribute. (Default = True).

        Returns
        -------
        critical_time_index : int
            Index of timepoint before critical curvature is first obtained
        """
        assert 0.0 <= crit <= 1.0, "Critical curvature value must be between zero and one."
        
        if self.kappas.max(axis=1).min() < crit:
            critical_time_index = np.where(self.kappas.max(axis=1) < crit)[0][-1]
        else:
            logger.info(f"Note: max curvature at time 0 is greater than critical value {crit}")
            critical_time_index = 0

        if update:
            kappa_critical = self.kappas.values[critical_time_index, :]
            nx.set_edge_attributes(self.G, dict(zip(self.kappas.columns.copy(), kappa_critical)), name="kappa_critical")

        return critical_time_index

    def critical_curvature_avg(self, crit=0.75, update=False):
        """ Return average curvature over critical time window

        Parameters
        ----------
        crit : float (0.0 <= crit <= 1.0) 
            Critical curvature value.
        update : boolean
            If True, saves critical curvatures as an edge attribute. (Default = True).

        Returns
        -------
        kappas_avg : pandas Series
            Series with average curvature for each edge in edgelist over critical time period.
        """
        assert 0.0 <= crit <= 1.0, "Critical curvature value must be between zero and one."
        
        t_crit = self.get_critical_timeindex(crit=crit, update=False)
        kappas_avg = self.kappas.iloc[0:t_crit+1, :].mean(axis=0)

        if update:
            nx.set_edge_attributes(self.G, kappas_avg.to_dict(), name="avg_kappa_critical")

        return kappas_avg.copy()

    def weighted_critical_curvature_avg(self, crit=0.75, update=False):
        """ Return weighted average curvature over critical time window

        Parameters
        ----------
        crit : float (0.0 <= crit <= 1.0) 
            Critical curvature value.
        update : boolean
            If True, saves critical curvatures as an edge attribute (wavg_kappa_critical). (Default = True).

        Returns
        -------
        kappas_wavg : pandas Series
            Series with weighted average curvature for each edge in edgelist over critical time period.
        """
        assert 0.0 <= crit <= 1.0, "Critical curvature value must be between zero and one."
        
        t_crit = self.get_critical_timeindex(crit=crit, update=False)
        tau_crit = self.times[t_crit]
        dt = np.ediff1d(self.times[:t_crit+1], to_begin=self.times[0])
        
        kappas_wavg = self.kappas.iloc[:t_crit+1, :].transpose().mul(dt).sum(axis=1).div(tau_crit)
        
        if update:
            nx.set_edge_attributes(self.G, kappas_wavg.to_dict(), name="wavg_kappa_critical")

        return kappas_wavg.copy()


    def time_to_critical_curvature(self, crit=0.75, update=False):
        """ Return time it took to get to critical curvature

        Parameters
        ----------
        crit : float (0.0 <= crit <= 1.0) 
            Critical curvature value.
        update : boolean
            If True, saves time to critical curvatures as an edge attribute. (Default = True).

        Returns
        -------
        titc : pandas Series
            Series with time index to reach critical curvature for all edges in edgelist.
        """
        assert 0.0 <= crit <= 1.0, "Critical curvature value must be between zero and one."

        def ix_ttc(x):
            """ time for curvature to reach critical value """
            if x.max() >= crit:
                ix = np.searchsorted(x, crit)
            else:
                ix = len(x) - 1
            return ix
        ti_crits = np.apply_along_axis(ix_ttc, 0, self.kappas)
        t_crits = self.times[ti_crits]
        
        if update:
            nx.set_edge_attributes(self.G, dict(zip(self.kappas.columns.copy(), ti_crits[:])),
                                   name="TiTC")  # time index
            nx.set_edge_attributes(self.G, dict(zip(self.kappas.columns.copy(), t_crits[:])),
                                   name="TTC")  # simulated time

        return pd.Series(data=ti_crits.copy(), index=self.kappas.columns.copy(), name="TiTC")
    
    
    def plot_dynamic_curvatures(self,
                                ylog=False,
                                ax=None,
                                figsize=(5, 4),
                                ):
        """ Plot simulated edge curvature evolution """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
        else:
            fig = None

        for kappa in self.kappas.values.T:
            if all(kappa > 0):
                color = "olive"  
            else:
                color = "tan"  
            ax.plot(np.log10(self.times), kappa, c=color, lw=0.5)

        if ylog:
            ax.set_xscale("symlog")
        ax.axhline(0, ls="--", c="k")
        ax.axis([np.log10(np.max([0, self.times[0]])), np.log10(self.times[-1]), np.min(self.kappas.values), 1])
        ax.set_xlabel(r"$log_{10}(t)$")
        ax.set_ylabel("Edge curvatures")

        return fig, ax

    def plot_dynamic_curvature_variance(self,
                                        ax=None,
                                        figsize=(5, 4),
                                        ):
        """Plot the variance of the curvature across edges."""

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
        else:
            fig = ax.get_figure()  

        ax.plot(np.log10(self.times), np.std(self.kappas.values, axis=1))
        ax.set_xlabel(r"$log_{10}$(t)")
        ax.set_ylabel("Edge curvature variance")
        ax.set_xlim([np.log10(self.times[0]), np.log10(self.times[-1])])

        return fig, ax
