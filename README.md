# FedMultimodal - 2023 KDD ADS
#### FedMutimodal [[Paper Link](https://arxiv.org/pdf/2306.09486.pdf)] is an open source project for researchers exploring multimodal applications in Federated Learning setup. FedMultimodal was accepted to 2023 KDD ADS track. 

The framework figure:

<div align="center">
 <img src="fed_multimodal/img/FedMultimodal.jpg" width="750px">
</div>


## Applications supported
* #### Cross-Device Applications
    * Speech Emotion Recognition
    * Multimedia Action Recognition
    * Human Activity Recognition
    * Social Media
* #### Cross-silo Applications (e.g. Medical Settings)
    * ECG classification
    * Ego-4D (To Appear)
    * Medical Imaging (To Appear)

### Installation
To begin with, please clone this repo:
```
git clone git@github.com:usc-sail/fed-multimodal.git
```

To install the conda environment:
```
cd fed-multimodal
conda create --name fed-multimodal python=3.9
conda activate fed-multimodal
```

Then pip install the package:
```
pip install -e .
```

### Quick Start -- UCI-HAR Example
Here we provide an example to quickly start with the experiments, and reproduce the UCI-HAR results from the paper. We set the fixed seed for data partitioning, training client sampling, so ideally you would get the exact results as reported from our paper.


#### 0. Download data: The data will be under data/uci-har by default. 

You can modify the data path in system.cfg to the desired path.

```
cd data
bash download_uci_har.sh
cd ..
```

#### 1. Partition the data

alpha specifies the non-iidness of the partition, the lower, the higher data heterogeneity.

```
python3 features/data_partitioning/uci-har/data_partition.py --alpha 0.1 --num_clients 5
python3 features/data_partitioning/uci-har/data_partition.py --alpha 5.0 --num_clients 5
```

#### 2. Feature extraction

For UCI-HAR dataset, the feature extraction mainly handles normalization.

```
python3 features/feature_processing/uci-har/extract_feature.py --alpha 0.1
python3 features/feature_processing/uci-har/extract_feature.py --alpha 5.0
```


#### 3. (Optional) Simulate missing modality conditions

default missing modality simulation returns missing modality at 10%, 20%, 30%, 40%, 50%

```
cd features/simulation_features/uci-har
# output/mm/ucihar/{client_id}_{mm_rate}.json

# missing modalities
bash run_mm.sh
cd ../../../
```

#### 4. Run base experiments (FedAvg, FedOpt, FedProx, ...)
```
cd experiment/uci-har
bash run_base.sh
```

#### Results for executing the above
Dataset | Modality | Paper | Label Size | Num. of Clients | Split | Alpha | FL Algorithm | Best UAR (Federated) | Learning Rate | Global Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| :---:| :---:|
UCI-HAR | Acc+Gyro | [UCI-Data](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | 6 | 105 | Natural+Manual | 5.0 <br> 5.0 <br> 0.1 <br> 0.1 |  FedAvg <br> FedOpt <br> FedAvg <br> FedOpt | 77.74% <br> 76.66% <br> 85.17% <br> 79.80% | 0.05 | 200 |



Feel free to contact us!

Tiantian Feng, University of Southern California

Email: tiantiaf@usc.edu