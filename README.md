# learning to segment

Pytorch implementation of the paper [Learning to segment images with classification labels
](https://arxiv.org/abs/1912.12533).

### *dependencies*:
```python
python==3.7
torch==1.0
tqdm==4.31
staintools==1.0
```

### *data*:

ICIAR 2018 BACH
Gleason 2019
DigestPath2019

Use create_seg_and_cls_sets.py for generating training/val data 
suitable for the model.

### *training*: 

use train_seg_and_cls_gleason.py for running the training script.

### *evaluation*:

run the experiments_displaystats.py for bach.

use the export_tensorboard_ynet.py for gleason2019 and digestpath2019.