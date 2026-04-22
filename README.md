# Overview

This repository contains the code used in the manuscript “Self-supervised Reservoir Computing with Spatial-temporal Encoding for Identifying Critical Transition”
submitted to Nature Communications.

# Repository Structure

    ├── Data/
      ├── Random_Wr_generation.py          # Generating random Wr of reservoir networks
      └── Simulation_data_generation.py    # Generating simulation data
    ├── main/
      ├── func_perf_assess.py              # functions of evaluating stARC
      ├── func_stARC_gpu.py                # functions of stARC
      └── main_stARC_gpu.py                # Main execution pipeline
    ├── README.md
    └── LICENSE

# Usage

1. Run `Random_Wr_generation.py` to generate generate and store the reservoir weight matrices $W_r$, which will be loaded automatically by subsequent scripts.
2. Run `Simulation_data_generation.py` to generate simulation data, including coupled Lorenz system, coupled Henon map and coupled ADVP system. For real-world dataset applied in this manuscription, the raw data of Stalagmite DP1 from Dante Cave used in this study is available from the National Centers for Environmental Information [https://www.ncei.noaa.gov/access/paleo-search/study/14268]. The source data of fish community data is available from EDI Data Portal [https://doi.org/10.6073/PASTA/C08B808FE0CA65AE30B65B7D780F037F]. The raw data of sediment cores in the Chew Bahir basin is available from zenodo repository [https://zenodo.org/records/10624471]. The source data of chicken heart data is available in github repository [https://github.com/ThomasMBury/dl_discrete_bifurcation].
3. Run `main_stARC_gpu.py` to apply the stARC framework to any target dataset.
