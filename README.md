# Overview

This repository contains the code used in the manuscript “Self-supervised Reservoir Computing with Spatial-temporal Encoding for Identifying Critical Transition”
submitted to Nature Communications.

# Repository Structure

    ├── Data/
      ├── Random_Wr_generation.py          # Generating random Wr of reservoir networks
      ├── Peter_lake.csv                   # Fish community structure data used in research 
      └── Simulation_data_generation.py    # Generating simulation data
    ├── main/
      ├── environment.yml                  # environment
      ├── func_perf_assess.py              # functions of evaluating stARC
      ├── func_stARC_gpu.py                # functions of stARC
      └── main_stARC_gpu.py                # Main execution pipeline
    ├── README.md
    └── LICENSE

# Usage
## Installation
1. Clone the repository
```bash
git clone <repository-url>
cd JetBrains
```
2. Create conda environment
```bash
conda env create -f environment.yml
conda activate yn_torch
```
## Running Experiments
1. Run `Random_Wr_generation.py` to generate generate and store the reservoir weight matrices $W_r$, which will be loaded automatically by subsequent scripts.
2. Run `Simulation_data_generation.py` to generate simulation data, including coupled Lorenz system, coupled Henon map and coupled ADVP system. For real-world dataset applied in this manuscription, the raw data of Stalagmite DP1 from Dante Cave used in this study is available from the National Centers for Environmental Information [https://www.ncei.noaa.gov/access/paleo-search/study/14268]. The source data of fish community data is available from EDI Data Portal [https://doi.org/10.6073/PASTA/C08B808FE0CA65AE30B65B7D780F037F]. The raw data of sediment cores in the Chew Bahir basin is available from zenodo repository [https://zenodo.org/records/10624471]. The source data of chicken heart data is available in github repository [https://github.com/ThomasMBury/dl_discrete_bifurcation].
3. Run `main_stARC_gpu.py` to apply the stARC framework to any target dataset.

## System Requirements
### Software
- **Python**: 3.10.x
- **PyTorch**: 2.8.0 (CUDA 12 support recommended
- Operating System: Linux (primary development platform)
- Dependencies: NumPy, SciPy, Pandas, Matplotlib, scikit-learn
### Hardware
| Minimum:
  - 8 GB RAM
  - x86_64 CPU (4+ cores
 Recommended for optimal performance:
  - NVIDIA GPU with CUDA 12.0+
  - 8+ GB VRAM
  - 16 GB system RAM

## Performance
Typical runtime on NVIDIA RTX 3090: 5-15 minutes per experiment
Memory footprint: ~3 GB GPU memory

## Reference
If you use this code in your research, please cite the corresponding paper.

## License
This code is provided for research purposes only.
