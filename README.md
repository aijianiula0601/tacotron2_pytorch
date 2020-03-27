# tacotron2_pytorch
make pytorch tactron2 train parallel with DataParallel

This repository is modified from https://github.com/BogiHsu/Tacotron2-PyTorch.

The nvida pytorch_tacotron2 training the model distribute with Apex.It is hard to debug and it only can use batch_size=8 when in Tesla V100 with 32G each Gpu.So I want to improve the training performance.

The codes with nvida has bug when modify it with DataParallel.The major mistake is the function:utils/util/get_mask_from_lengths.I fixed it and can train with batch_size=32 in each gpu in Telsa V100.The result similary to https://github.com/BogiHsu/Tacotron2-PyTorch.
