# Scan2Cap: Generate Descriptions for 3D Scans 

## Lik to Video report
Our final presentation can be found on [here](https://www.youtube.com/watch?v=RB6qlZPY2iM). 

## Setup

Setup conda environment:
```shell
conda env create --file env/environment.yml
```

Compile PoinNet++:
```shell
cd scan2cap/lib/pointnet2
python setup.py install
```

## Execution

Pretraining of PointNet++:
```shell
python scripts/train_pretrain.py
```

Training: 
```shell
python scripts/train_scan2cap.py
```

Visualize Results: 
```shell
python scripts/visualize_scan2cap.py
```

Visualize Results generated w_o GT: 
```shell
python scripts/visualize_scan2cap_nogt.py
```
Please refer to the scripts for detailed explanation of the flags.


## Acknowledgement
We would like to thank [daveredrum/ScanRefer](https://github.com/daveredrum/ScanRefer) for providing the training framework for mixed 3D CV/ NLP tasks on ScanRefer and for the valuable supervision.

We would like to thank [facebookresearch/votenet](https://github.com/facebookresearch/votenet) for the 3D object detection codebase and [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) for the CUDA accelerated PointNet++ implementation.

We would like to thank [sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning ](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/tree/71dd0ca353ce7a2373177eb1c798cda05db36ff8) for providing the inspiration for that captioning task.