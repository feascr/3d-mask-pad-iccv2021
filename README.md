# 3D High-Fidelity Mask Face Presentation Attack Detection Challenge

## Train models
To reproduce results you need to specify `data_path` and `save_path` in `./configs/pretrain_config.py` and
start training process:
```bash
python train.py --config ./configs/pretrain_config.py
```

After pretraining stage is finished you need to specify `data_path`, `save_path` and `backbone.pretrained_path` (checkpoint with the lowest ACER on dev set was chosen for pretrained weights) in `./configs/train_config.py` then start training:
```bash
python train.py --config ./configs/train_config.py
```

## Inference model
To get val_test.txt specify `data_path`,  `result_directory` and  `checkpoint_path` (checkpoint with the lowest ACER on validation was chosen) in `./configs/pretrain_config.py` and
start evaluating process:
```bash
python evaluate.py --config ./configs/evaluate.py
```