# HyperspecAE
This repository contains the pytorch implementation for the paper: [Hyperspectral Unmixing Using a Neural Network Autoencoder](https://ieeexplore.ieee.org/document/8322133) (Palsson et al. 2018). As a POC, the dataloaders and parameters corresponding to experiments on the Samson dataset are presented. 

# Dependencies
* __PyTorch 1.8.0__
* __Python 3.7.10__

# Quick Start
## Data
The datasets used in the paper are publicly available and can be found [here](https://rslab.ut.ac.ir/data).<br>
Download the Samson dataset from the above-mentioned source. Follow the directory tree given below:<br>
```
|-- [root] HyperspecAE\
    |-- [DIR] data\
        |-- [DIR] Samson\
             |-- [DIR] Data_Matlab\
                 |-- samson_1.mat
             |-- [DIR] GroundTruth
                 |-- end3.mat
                 |-- end3_Abundances.fig
                 |-- end3_Materials.fig
```

## Training
The shell script that trains the model (```samson_train.sh```) can be found in the [run folder](https://github.com/dv-fenix/HyperspecAE/tree/main/run). You can simply alter the hyperparameters and other related model options in this script and run it on the terminal.<br>
You can refer to [opts.py](https://github.com/dv-fenix/HyperspecAE/blob/main/src/utils/opts.py) to explore other command line arguments to customize model training.

## Abundance Map and End-Member Extraction
The shell script that extracts the abundance maps and end-members (```extract.sh```) can be found in the [run folder](https://github.com/dv-fenix/HyperspecAE/tree/main/run). Ensure that the charateristics of the model match exactly with the pre-trained version to be used for extraction.<br>

# Results
The following are the results of a deep autoencoder, Configuration Name: LReLU (see paper). You can experiment with other configurations by altering the command line arguments during model training.

## Pre-trained Model
The pre-trained model for this configuration can be found [here](https://github.com/dv-fenix/HyperspecAE/tree/main/logs).

## Abundance Maps
![abundances](https://user-images.githubusercontent.com/45421556/112638558-627afb00-8e65-11eb-8bd2-3669313f7e5a.png)
**Left**: Tree, **Middle**: Water and **Right**: Rock.

## Extracted Spectral Signature (End-Members)
![end_members](https://user-images.githubusercontent.com/45421556/112639814-b1756000-8e66-11eb-84f3-04151e6665a2.png)
**Left**: Tree, **Middle**: Water and **Right**: Rock.

# References
Original work by the authors
```
@article{palsson2018hyperspectral,
  title={Hyperspectral unmixing using a neural network autoencoder},
  author={Palsson, Burkni and Sigurdsson, Jakob and Sveinsson, Johannes R and Ulfarsson, Magnus O},
  journal={IEEE Access},
  volume={6},
  pages={25646--25656},
  year={2018},
  publisher={IEEE}
}
```
