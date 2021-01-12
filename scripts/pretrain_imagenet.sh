#!/bin/bash

SERVER=$(echo $1 | tr '[a-z]' '[A-Z]')

if [ ${SERVER} == "LOCALHOST" ]; then
  echo "[INFO] The server specified as ${SERVER}"
  DATA_ROOT="/media/dandri/DANDRI/ILSVRC2012"
  OUTPUT_ROOT="../output"
  PYTHON_ENV="/home/dandri/anaconda3/envs/pytorch/bin/python"
elif [ ${SERVER} == "KB541-TITAN" ]; then
  echo "[INFO] The server specified as ${SERVER}"
  DATA_ROOT="/media/E/DCJ_DATA/Datasets/ILSVRC2012"
  OUTPUT_ROOT="/media/E/DCJ_DATA/Output/ILSVRC2012"
  PYTHON_ENV="/home/dengchangjian/.conda/envs/pytotch-gpu/bin/python"
elif [ ${SERVER} == "KB541-RTX2080TI" ]; then
  echo "[INFO] The server specified as ${SERVER}"
  DATA_ROOT="/media/lab/DANDRI/Datasets/Public/ILSVRC2012"
  OUTPUT_ROOT="/media/hdd1/dcj_data/output/ILSVRC2012"
  PYTHON_ENV="/usr/local/miniconda3/envs/pytorch-1.6/bin/python"
else
  echo "[ERROR] Please specify the server"
  exit
fi

if [ ! -x "$PYTHON_ENV" ]; then
  echo "[ERROR] Please change {PYTHON_ENV} to your conda environment before running"
  exit
fi

${PYTHON_ENV} --version

${PYTHON_ENV} ../pretrain.py \
  --train_size 224 \
  --valid_size 256 \
  --train_batch 512 \
  --valid_batch 250 \
  --num_epoch 180 \
  --num_workers 16 \
  --optimizer 'SGD' \
  --init_lr 0.1 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --milestones '30,60,90,120,150' \
  --milestones_gamma 0.1 \
  --phase "train" \
  --seed 369 \
  --model "GhostNet1p3" \
  --dataset "imagenet" \
  --num_class 1000 \
  --data_root "${DATA_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --gpus "0,1,4,5" \
  --resume "epoch_last.pth" \
  --dali_gpu \
  --prefetch 2
