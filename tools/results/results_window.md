
##  MAE-Baselines:

### wdw-size 5:
train-dev 1.4299739961677862  
train-test 1.0341010816006755  

### wdw-size 10:
train-dev 2.7532740775494466  
train-test 1.991912320916103  

### wdw-size 20:
train-dev 5.479152638507392  
train-test 4.075823361771275  

## Trained on folder 0: on WvLmCNN14...

    fold 0: trained on best (wvlmcnn14)

    different windows of length 5 second:
    ____________________________________________________
    window       CC      CCC     MSE     MAE    epoch
    0.0-4.17    0.706   0.696   1.187   0.506  43/60
    0.0-5.12    0.432   0.420   2.042   0.799  24/60
    0.0-5.00    0.484   0.442   2.684   0.951  19/80
    ____________________________________________________

    window-length:
    ____________________________________________________
    size       CC      MSE     MAE              Baseline-MAE      epoch
    5s         0.706   1.187   0.506(+50%)      1.034             25/100
    10s        0.599   7.394   1.577(+21%)      1.991             24/100=
    20s
    ____________________________________________________




-----


[//]: # (### old experiment:)

[//]: # ()
[//]: # (    mae                 CC      hparam                   architecture   [arch]      epoch      improvement     time)

[//]: # ()
[//]: # (    3.5577423572540283  0.36    Adam_b32_1e-3            cnn10          [cnn]       20/20      12.7111 % )

