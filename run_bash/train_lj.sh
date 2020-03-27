#!/bin/bash

set -ex

#-------------------------------------------------------------
#说明：
#项目参考：https://github.com/BogiHsu/Tacotron2-PyTorch
#这个提取的音频的采用率为22050,推断时候注意修改超参数，per_step=2
#-------------------------------------------------------------

cd ../

day_str=`date +%Y%m%d-%H`

python -V

base_dir='/home/huangjiahong/tmp/tts/dataset/api/real_time_voice_clone_dataset/ljspeech'
meta_file_path="${base_dir}/train.txt"
log_dir="${base_dir}/tacotron2_pytorch_bogihsu_v2/${day_str}/log"
checkpoint_dir="${base_dir}/tacotron2_pytorch_bogihsu_v2/${day_str}/checkpoints"

mkdir -p $log_dir
mkdir -p $checkpoint_dir


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u train.py \
--meta_file $meta_file_path \
--log_dir $log_dir \
--ckpt_dir $checkpoint_dir