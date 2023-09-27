
# Different checkpoints/pretrain weights, same models/input:

sources:
https://github.com/kkoutini/PaSST - passt
https://github.com/YuanGongND/psla#Pretrained-Models - psla
https://github.com/qiuqiangkong/audioset_tagging_cnn - panns

## Fold 0, WDW5:
    psla:
        CC      CCC     MSE     MAE     pretrain    @ epoch
        0.647   0.598   1.357   0.608   imagenet    10  
        0.586   0.559   1.567   0.631   no          30
        0.637   0.611   1.668   0.751   fsdk50k     20
        0.591   0.577   2.064   0.807   audioset    20             

