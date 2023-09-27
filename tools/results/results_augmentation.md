
# Different augmentations, same models:

Results on trying different augmentations on the same architectures.  
Sorted by MAE descending

## Fold 0, WDW5:
    cnn10:
        CC      CCC     MSE     MAE     augment           epochs
        0.552   0.447   1.605   0.639   filteraug         50
        0.560   0.503   1.580   0.640   specaug           50
        0.550   0.494   1.614   0.643   none              50
        0.549   0.464   1.606   0.668   mixup@0.3         50
        0.513   0.408   1.692   0.675   mixup@1.0         50
        0.509   0.430   1.688   0.690   mixup@0.7         50
        0.426   0.315   1.864   0.776   mixup@0.1         50

    wvlmcnn14(imnet pretrain):
        CC      CCC     MSE     MAE     augment           epochs

        0.706   0.696   1.187   0.506   None              60
        0.666   0.647   1.314   0.563   FiltAug           100  
        0.743   0.720   1.132   0.609   SpecAug           100
        0.631   0.561   1.609   0.793   Mixup             30


