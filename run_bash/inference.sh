#!/bin/bash

set -ex

#-------------------------------------------------------------
#说明：
#项目参考：https://github.com/BogiHsu/Tacotron2-PyTorch
#-------------------------------------------------------------

cd ../

python -V


checkpoint_path='/home/huangjiahong/tmp/tts/dataset/api/real_time_voice_clone_dataset/ljspeech/tacotron2_pytorch_bogihsu_v2/20200324-15/checkpoints/ckpt_16500'
image_path='/tmp/tpb_img'
wav_path='/tmp/tpb_test'
npy_path='/tmp/tpb_test'
text=' The first project of putting forward to use the shallow random CNN for voice transfer.'

CUDA_VISIBLE_DEVICES=7 \
python -u inference.py \
--ckpt_pth $checkpoint_path \
--img_pth $image_path \
--wav_pth $wav_path \
--npy_pth $npy_path \
--text "$text"
