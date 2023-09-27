
Evenly sized, 5.0-second windows.
Results Sorted by MAE descending.

    # train - dev 1.4442569338090476
    # train - test 1.1003917548651998

## fold0, 5s:

    fold 0: MAE between train and test:  1.1004
    model       CC      CCC     MSE     MAE         input       time    epoch   arch        pretrain

    wvlmcnn14:  0.528   0.485   2.371   0.847       wavegram    3h/12h  54/100  [CNN]       -    
    psla:       0.438   0.423   2.856   0.905
    passt:      0.419   0.403   2.943   0.935       melspec     /20h
    fnn:        0.000   0.000   3.102   0.946       mfcc,7      30m/1h  10/20   [DNN]
    hts-at:     0.054   0.001   3.154   0.959                           
    base:       -       -       -       1.100       -           -       -       -           -




## fold0, 10s:

    fold 0: MAE between train and test:  1.1004
    model       CC      CCC     MSE     MAE         input       time    epoch   arch        pretrain

    wvlmcnn14:  0.528   0.485   2.371   0.847       wavegram    3h/12h  54/100  [CNN]       -    
    passt:      0.419   0.403   2.943   0.935       melspec     /20h

    base:       -       -       -       1.100       -           -       -       -           -

## fold0, 20s:

    fold 0: MAE between train and test:  1.1004
    model       CC      CCC     MSE     MAE         input       time    epoch   arch        pretrain

    base:       -       -       -       4.089      -           -       -       -           -

## CROSS FOLD:

    model       CC      CCC     MSE     MAE         input       pretrain
    s0+
    psla        0.534   0.486   2.270   0.814                   ImageNet
    wvlmcnn14:  0.528   0.485   2.371   0.847                   audioset
    passt:      0.419   0.403   2.943   0.935
    base:                               1.100

    s1
    psla        0.536   0.494   2.951   1.009                   ImageNet
    wvlmcnn14:  0.478   0.433   3.159   1.090                   audioset
    passt:      0.482   0.374   3.074   1.145
    base:                               1.435

    s2
    wvlmcnn14:  0.422   0.345   5.008   1.277                   -
    passt:      0.410   0.352   5.491   1.319                    (<100)
    psla        0.288   0.215   6.068   1.398                   ImageNet
    base:                               1.531

    s3          
    psla        0.383   0.285   5.526   1.225                   ImageNet
    passt:      0.453   0.309   5.248   1.260                    (<100)
    wvlmcnn14:  0.390   0.248   5.437   1.492                    (<100)
    base:                               1.548

    s4
    wvlmcnn14:  0.578   0.432   4.291   1.153
    psla        0.472   0.368   5.145   1.219                   ImageNet
    passt:      0.451   0.369   5.108   1.273                    (<100)
    base:                               1.492

    mean:
    wvlmcnn14:  0.479   0.388   4.053   1.171
    passt:      0.443   0.361   4.372   1.186
    baseline:                           1.421

# Transfer Learning (psla)

    CC      CCC     MSE     MAE         pretrain

    0.534   0.486   2.270   0.814       ImageNet
    0.459   0.441   2.730   0.875       AudioSet
    0.443   0.434   2.973   0.879       Random
    0.320   0.271   2.964   0.911       FSD50K










