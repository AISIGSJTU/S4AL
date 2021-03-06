# S4AL: Self-Supervised Vessel Segmentation with Synthetic Medical Images via Adversarial Learning

This project is the PyTorch implementation of paper "S4AL: Self-Supervised Vessel Segmentation with 
Synthetic Medical Images via Adversarial Learning".

## I. Environment Configuration

Follow `resources/env.yml` to create the conda virtual environment: `conda env create -f env.yml`

## II. BRM

`python brm.py`

## III. DRL-MSM

### Data Preparation

Run `generate_raw_fractals.py` to generate the following folders with data:

- `datasets/rl/train`

- `datasets/rl/test`

Run `generate_data4predictor.py` to generate the following folders with data:

- `datasets/predictor/train`

- `datasets/predictor/test`

### S-FID Predictor

0. Download the **Vessel Mask Set** from [Google Drive](https://drive.google.com/file/d/1yhjizXPefrDqJPdYO5lDGp9VEdmCPdhk/view?usp=sharing) 
or [Baidu Netdisk](https://pan.baidu.com/s/1fykmJltJPOW4sUcqxqIhQw) (key：___euzs___). Extract and put the folder under 
`datasets/target`.

0. Following the paper to calculate S-FID for each image. We recommend 
[PyTorch-FID](https://github.com/mseitzer/pytorch-fid) for the calculation of FID. 
Save the results in `datasets/predictor/train/fid_train.txt`, following the format as in `resources/demo.txt`.

0. Train the S-FID predictor by supervised learning: `python train_predictor.py`.

0. We provide a trained model for the S-FID predictor, saved in 
`msm/predictor/mobilenet/predictor.tar`, which can directly be used for the 
training of the policy model.

### Train the Policy Model
`python train_policy.py --predictor-ckp msm/predictor/mobilenet/predictor.tar --seed 79 --memory-size 500000 --random-steps 10000 --batch-size 512 --gamma 0.9 --lr-actor 1e-4 --lr-critic 3e-4`

## IV. Vessel Representations Learning

### Data Preparation

You can download the XCAD dataset from [Dropbox](https://www.dropbox.com/s/z0lk5oz6gt9mgd2/XCAD.zip?dl=0) or 
[Baidu Netdisk](https://pan.baidu.com/s/1C9d9_92TSDBGBfagatTpoA) (key: ___neia___).

0. Put the downloaded dataset to dataset/ssv.

    - `XCAD/train/*` -> `datasets/ssv/*`
    - `XCAD/test/images` -> `datasets/ssv/testB`
    - `XCAD/test/masks` -> `datasets/ssv/testA`

0. Put some background images to `datasets/ssv/testC`.

0. Put fake masks generated by DRL-MSM in `datasets/ssv/trainA`.

### Train

`CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataroot datasets/ssv --gpu_ids 0,1,2,3 --name check --lr 0.00001 --lr_policy multistep --model usseg --dataset_mode usseg --display_env check --n_epochs 30 --n_epochs_decay 0 --save_epoch_freq 5 --batch_size 4 --no_flip --lambda_A 10 --lambda_B 10`

### Test

`python test.py --dataroot dataset/ssv --name check --model usseg --dataset_mode usseg --epoch 25`
