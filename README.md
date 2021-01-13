# LightWeight-HRRSI

Semantic Segmentation for High-Resolution Remote Sensing Images by Light-Weight Network



## Briefly

* This repo introduces a light-weight semantic segmentation network for embedding platforms, such as micro aerial vehicles (MAVs)
* The network only **requires 2.83M parameters and 2.37GFLOPs** for inference
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

# Install packages and dependencies
pip install torch==1.6.0 torchvision==0.8.0 [-i https://pypi.douban.com/simpe]
pip install imgaug matplotlib tqdm prettytable [-i https://pypi.douban.com/simpe]
```



## Models

- All the models involved in the experiment are in folder `models/`

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

  * **UED**：model in `seg_ghostnet.py` , width_multi=1.0
  * **UED+BES**：model in `seg_ghostnet_decouple.py` , width_multi=1.0
  * **UED+BES+SFM**: model in `seg_ghostnet_decouple_score.py` , width_multi=1.0

* Under the condition of the image size is 512x512, the performances of the above models on the Vaihingen dataset are as follows:

  | Model       |  mF1  | mIoU  |  OA   | Params(M) | FLOPs(G) |
  | :---------- | :---: | :---: | :---: | :-------: | :------: |
  | UED         | 87.33 | 77.91 | 88.67 |   2.77    |   1.44   |
  | UED+BES     | 87.83 | 78.69 | 89.25 |   2.82    |   2.19   |
  | UED+BES+SFM | 88.19 | 79.27 | 89.60 |   2.83    |   2.37   |

  

## Usage

* It is recommended to start training or testing from `scripts/`. Let's take the training phase as an example,

  * Modify the `DATA_ROOT`, `OUTPUT_ROOT`, and `PYTHON_ENV`  in **train_vaihingen.sh**

  * Set the hyper-parameters for training phase, such as `learning rate`, `weight decay`

  * Specify the models you need to train, e.g.

    ```shell
    ${PYTHON_ENV} ../main.py \
      ...
      --model "SegGhostNet1p0,SegGhostNetDecouple1p0,SegGhostNetDecoupleScore1p0" \
      ...
    ```

  * Run the script in terminal (There's no need to activate your conda environment)

    ```shell
    ./scripts/train_vaihingen.sh localhost
    ```

* In addition, the pre-training code on **ImageNet** is provided, and the usage is similar to **train_vaihingen.sh**. The pre-training process is accelerated by [DALI](https://github.com/NVIDIA/DALI) library. It's worth noting that you need to prepare your training and validation datasets in advance in the following way.

  ```
  .
  |-- train
  |   |-- n01440764
  |   |-- ...
  |-- val
  |   |-- n01440764
  |   |-- ...
  ```



## TODO

* Model pruning
* Quantization compression
* Deployment and application on embedding platforms

