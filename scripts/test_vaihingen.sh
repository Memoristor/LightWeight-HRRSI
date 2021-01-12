#!/bin/bash

SERVER=$(echo $1 | tr '[a-z]' '[A-Z]')

if [ ${SERVER} == "LOCALHOST" ]; then
  echo "[INFO] The server specified as ${SERVER}"
  DATA_ROOT="/home/dandri/data/vaismall"
  OUTPUT_ROOT="../output"
  PYTHON_ENV="/home/dandri/anaconda3/envs/pytorch/bin/python"
elif [ ${SERVER} == "KB541-TITAN" ]; then
  echo "[INFO] The server specified as ${SERVER}"
  DATA_ROOT="/media/E/DCJ_DATA/Datasets/Vaihingen"
  OUTPUT_ROOT="/media/E/DCJ_DATA/Output/IGRASS-2021"
  PYTHON_ENV="/home/dengchangjian/.conda/envs/pytotch-gpu/bin/python"
elif [ ${SERVER} == "KB541-RTX2080TI" ]; then
  echo "[INFO] The server specified as ${SERVER}"
  DATA_ROOT="/media/ssd/dcj_data/data/vaismall"
  OUTPUT_ROOT="/media/hdd1/dcj_data/output/IGRASS-2021"
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

${PYTHON_ENV} ../main.py \
  --input_size 512 \
  --num_epoch 500 \
  --batch_size 8 \
  --num_workers 8 \
  --optimizer 'SGD' \
  --init_lr 0.004 \
  --lr_gamma 0.9 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --phase "test" \
  --seed 369 \
  --model "SegGhostNet1p0,SegGhostNetDecouple1p0,SegGhostNetDecoupleScore1p0" \
  --dataset "vaihingen" \
  --data_root "${DATA_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --gpus "1" \
  --resume "<best:OA>"
