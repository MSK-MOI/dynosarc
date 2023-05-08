# Article 
The preprint of the manuscript "Dynamic Network Curvature Analysis of Gene Expression Reveals Novel Potential Therapeutic Targets in Sarcoma" can be found at 
[https://doi.org/10.1101/2022.03.09.483487](https://doi.org/10.1101/2022.03.09.483487)

## Cite

Please cite our paper if you use this code in your work
```
@article{elkin2023geometry,
  title={Geometry of gene expression network reveals potential novel indicator in Ewing sarcoma},
  author={Elkin, Rena and Oh, Jung Hun and Dela Cruz, Filemon and Norton, Larry and Deasy, Joseph O and Kung, Andrew L and Tannenbaum, Allen R},
  journal={Cancer Research},
  volume={83},
  number={7\_Supplement},
  pages={6541--6541},
  year={2023},
  publisher={AACR}
}
```

# Introduction
__DYNOsaRC__ is a general purpose package for performing Wasserstein-based hierarchical clustering and computing dynamic Ollivier-Ricci curvature on weighted graphs.

Examples running the code to reproduce the analysis performed in the manuscript can be found in the `notebooks` folder.
Briefly, DYNOsaRC analysis was applied to analyze weighted transcriptomic networks associated with pediatric sarcomas. The two-fold analysis entailed:
1. Performing Wasserstein-based subtyping 
<p align="center">
<img src="/figures/Wass_subtype_clustering_heatmap.png" width="500">
</p>  

2. Analyzing persistent functional gene associations via dynamic Ollivier-Ricci curvature 
<p align="center">
<img src="/figures/dyno_EWS_ms_persistent_clustering.png" width="800">
</p>  

Details of the approach can be found in the manuscript. 

# Dependencies
* NetworkX
* NumPy
* pandas
* Pot
* SciPy

# Acknowledgements

The code for computing dynamic Ollivier-Ricci curvature was largely based off of the code written by Gosztolai and Arnaudon
```
@article{GosztolaiArnaudon2021,
author = {Gosztolai, Adam and Arnaudon, Alexis},
doi = {10.1038/s41467-021-24884-1},
issn = {2041-1723},
journal = {Nat. Commun.},
number = {1},
pages = {4561},
title = {{Unfolding the multiscale structure of networks with dynamical Ollivier-Ricci curvature}},
volume = {12},
year = {2021}
}

```
