# BESNet-LightWeight-HRRSI

Semantic Segmentation for High-Resolution Remote Sensing Images by Light-Weight Network



## Briefly

* This repo introduces a light-weight semantic segmentation network for embedding platforms, such as micro aerial vehicles (MAVs).
* The network only **requires 2.83M parameters and 2.37GFLOPs** for inference.
* The overall accuracy (OA) of this light-weight network reaches **89.60%** on the validation set of the public Vaihingen dataset (tile ID 11, 15, 28, 30, and 34) 



## Environment

### Runtime Environment

- Ubuntu 18.04.5 LTS (8G RAM)
- PyTorch 1.6.0
- CUDA 10.2+
- NVIDIA RTX2080 Ti (11G)

### Conda Environment

```shell
# Create your conda environment
conda create -n pytorch python=3.6

# Activate conda environment and upgrade pip
conda/source activate pytorch
python -m pip install --upgrade pip

# Install dependencies
pip install torch==1.6.0 torchvision==0.8.0 [-i https://pypi.douban.com/simpe]
pip install imgaug matplotlib tqdm prettytable [-i https://pypi.douban.com/simpe]
```



## Models

```
.
|-- previous
|   |-- deeplabv3_plus.py
|   |-- ...
|-- seg_ghostnet.py
|-- seg_ghostnet_decouple.py
|-- seg_ghostnet_decouple_score.py
`-- seg_resnet.py
```

* 
* 



## Usage



