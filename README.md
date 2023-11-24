# deepDR
## deepDR: A network-based deep learning approach to in silico drug repositioning
> https://doi.org/10.1093/bioinformatics/btz418

> https://github.com/ChengF-Lab/deepDR

### 'dataset' directory
Contain the drug-disease interactions dataset.
### 'PPMI' directory
Contain the PPMI matrices of ten drug-related networks.
### 'models' directory
1. 'deep_network_fusion' directory, which contains 3 variants implementations of Multimodal Deep Autoencoder (MDA)
    > https://doi.org/10.1093/bioinformatics/bty440
2. 'recommendation' directory, which contains implementation of Collective Variational Autoencoder (cVAE)
    > https://doi.org/10.1145/3270323.3270326
    - Updated for calculating Recall@K metric.
### Requirements
...
### Tutorial
1. To get drug features learned by MDA, run
```
python get_features.py params.txt
```
2. To predict drug-disease associations by cVAE, run
   1. pretraining with features:
       ```
       python models/recommendation/collective_variational_autoencoder.py --dir dataset -a 6 -b 0.1 -m 300 --save --layer 1000 100
       ```
   2. refine training with rating:
       ```
       python models/recommendation/collective_variational_autoencoder.py --dir dataset --rating -a 15 -b 3 -m 500 --load 1 --layer 1000 100
       ```
- On colab, simply follow the 'deepDR.ipynb' notebook.