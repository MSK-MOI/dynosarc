import os
import itertools

import cvxpy as cp
import multiprocessing as mp
import networkx as nx
import numpy as np
import numpy.matlib
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp
from tqdm import tqdm

from dynosarc.gen_logger import gen_logger, set_verbose

logger = gen_logger(__name__)
set_verbose(logger, verbose="INFO")


def compute_invariant_measure(G, w):
    """ Compute the invariant distribution (invr) from given graph and node weights

    Parameters
    ----------
    G : NetworkX Graph
        The graph.
    w : Pandas Series
        Node weights.

    Returns
    ------
    invr : dict
        The computed invariant distribution returned as dict of the form {node x: invr_x}.
    """
    M = G.copy()
    nx.set_node_attributes(M, w.to_dict(), name='weight')

    if not nx.is_connected(G):
        raise TypeError("Network must be connected.")

    invr = {}
    for n in G.nodes():
        n_invr = M.nodes[n]['weight'] * sum([M.nodes[nbr]['weight'] for nbr in M.neighbors(n)])
        invr[n] = n_invr
    sum_invr = sum(invr.values())
    if sum_invr > 1e-7:
        invr = {k: val / sum_invr for k, val in invr.items()}
    else:
        raise Exception("Sum of invariant measure is too small to normalize.")

    return invr

def compute_invariant_measures_for_cohort(G, D, fname, verbose="WARNING"):
    """ compute and invariant distributions (invrs) for all samples in given dataset D

    Parameters
    ----------
    G : NetworkX Graph
        The graph.
    D : Pandas DataFrame
        Sample features used as node weights to compute the invariant distributions where rows are features and columns are samples.
    fname : str
        Filename where resulting invariant distributions will be saved.
    verbose : {'DEBUG', 'INFO', 'WARNING', 'ERROR'}
        Verbosity level (default="ERROR").

    Returns
    -------
    invrs : Pandas DataFrame
        Invariant distributions for each sample given as the columns in the dataframe.
    """
    set_verbose(logger, verbose=verbose)
    
    # compute invariant distributions
    if not os.path.isfile(fname):
        logger.info("Computing invariant distributions")
        invrs = {m: compute_invariant_measure(G.copy(), data) for m, data in D.iteritems()}
        invrs = pd.DataFrame.from_dict(invrs, orient='columns')
        invrs.to_csv(fname,header=True,index=True)
    else: # return file
        logger.info("Invariant distributions loaded from file")
        invrs = pd.read_csv(fname, sep=',', header=0, index_col=0)
    return invrs


def dist_cvx(drho,D,m):
    """ Compute Wasserstein distance from incidence matrix """
    # Create two scalar optimization variables.
    u = cp.Variable((m))
    # Form objective.
    objective = cp.Minimize(cp.sum(cp.abs(u)))
    # Create two constraints.
    constraints = [drho-D@u == 0]
    # Form and solve problem.
    problem = cp.Problem(objective, constraints)
    problem.solve()
    return problem.value


def invr_EMD_single_pair(i,j):
    """ EMD between invariant distributions """
    rho0 = _Node_Prob[:,i]
    rho1 = _Node_Prob[:,j]
    drho = rho0-rho1
    emd=dist_cvx(drho,_D,_m)
    return {(i,j): emd}
    

def wrap_compute_single_pair(stuff):
    """Wrapper for pairs in multiprocessing."""
    return invr_EMD_single_pair(*stuff)


def EMD_matrix(X, G, fname, proc=mp.cpu_count(), chunksize=None, verbose="ERROR"):
    """ compute pairwise Wasserstein distances between columns of X """

    set_verbose(logger, verbose)
    if not os.path.isfile(fname):
        logger.info("Computing EMD matrix")
        global _D
        global _Node_Prob 
        global _m
        adj = nx.convert_matrix.to_numpy_array(G, nodelist=list(X.index.copy()),
                                               dtype=np.float64, weight=None)
        no_sample = X.shape[1]
        no_feature = X.shape[0]

        _m = int(round(adj.sum()/2))

        _D = nx.linalg.graphmatrix.incidence_matrix(G, nodelist=list(X.index.copy()),
                                                    oriented=True, weight=None)
        Node_weights = X.values
        S = X.sum(axis=0) 
        S_copy = np.matlib.repmat(S,no_feature,1)
        _Node_Prob = Node_weights/S_copy
        pairs = list(itertools.combinations(list(range(no_sample)),2)) # args
        with mp.get_context('fork').Pool(processes=proc) as pool:
            if chunksize is None:
                chunksize, extra = divmod(len(pairs), proc * 4)
                if extra:
                    chunksize += 1

            result = pool.imap_unordered(wrap_compute_single_pair, pairs,
                                         chunksize=chunksize)
            pool.close()
            pool.join()
        Distance_Matrix =np.zeros((no_sample,no_sample))
        for emd in result:
            for k in list(emd.keys()):
                Distance_Matrix[k[0],k[1]] = emd[k]
        Distance_MatrixM = Distance_Matrix+np.transpose(Distance_Matrix)
        emds = pd.DataFrame(data=Distance_MatrixM, index=X.columns.copy(),
                            columns=X.columns.copy())
        emds.to_csv(fname)
    else:
        logger.info("EMDs loaded from file")
        emds = pd.read_csv(fname, sep=',', header=0, index_col=0)
    return emds




