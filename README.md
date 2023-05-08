# Article 
The preprint of the manuscript "Dynamic Network Curvature Analysis of Gene Expression Reveals Novel Potential Therapeutic Targets in Sarcoma" can be found at 
[https://doi.org/10.1101/2022.03.09.483487](https://doi.org/10.1101/2022.03.09.483487)

# Introduction
__dynosarc__ is a general purpose toolbox for performing Wasserstein-based hierarchical clustering and computing dynamic Ollivier-Ricci curvature on weighted graphs.

Examples running the code to reproduce the analysis performed in the manuscript can be found in the Notebooks folder.
Briefly, dynosarc analysis was applied to analyze weighted transcriptomic networks associated with pediatric sarcoma. The two-fold analysis entailed:
1. Performing Wasserstein-based subtyping 
<p align="center">
<img src="/figures/Wass_subtype_clustering_heatmap.png" width="500">
</p>  

2. Analyzing persistent functional gene associations via dynamic Ollivier-Ricci curvature 
<p align="center">
<img src="/figures/dyno_EWS_ms_persistent_clustering.png" width="800">
</p>  

Details of the approach can be found in the manuscript. 
