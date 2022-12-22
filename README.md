# Bi-stride Multi-Scale GNN

This repository contains implementations of the code for [Bi-stride Multi-Scale GNN](https://arxiv.org/abs/2210.02573).

## Requirements

- Pytorch
- PyG
- Numpy
- h5py
- TensorBoard
- SciPy
- scikit-learn
- sparse-dot-mkl

## Download datasets and pretrained models

We host the datasets and pretrained models on this [link](https://drive.google.com/drive/folders/15UjqYdDX_Zhf-uPIs0bIZ5hYVjNiZUZy?usp=share_link). Please keep the file structure as below to run the script by default.

```
this project
│   ...    
│
└───data
│   └───cylinder
|       └───outputs_test
|       └───outputs_train
|       └───outputs_valid
│       │   meta.json
│   └───...
└───res
│   └───cylinder
|       └───ours
|       |   └───ckpts
|       |       |   *.pt
│   └───...
```

If you store the data and result folders somewhere else, you can modify the `data_dir` and `dump_dir` in the config files correspondingly.

## How to use

```sh
# ./run_BSMS.sh $case_name ./configs/$case_name $mode $restart_epoch

# case_name: [cylinder, airfoil, plate, font]
# ./configs/$case_name: store the corresponding config files of a case
# mode: [0:train, 1:local test, 2: global rollout]
# restart_epoch: -1 (or leave blank) train from start; 0,1... reload the stored ckpts of certain frame

# e.g. train font from scratch
./run_BSMS.sh font ./configs/font 0 -1
# e.g. local test RMSE of cylinder at epoch 19
./run_BSMS.sh cylinder ./configs/cylinder 1 19
# e.g. global rollout RMSE of airfoil at epoch 39
./run_BSMS.sh airfoil ./configs/airfoil 2 39
``` 
