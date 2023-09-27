# 5 FOLD CROSS VALIDATION:

Results Sorted by MAE descending.

## Trained on folder 0:

    fold 0: MAE between train and test:  1.0341010816006755
    model       CC      CCC     MSE     MAE         input       time    epoch   arch        pretrain

    wvlmcnn14:  0.706   0.696   1.187   0.506       wavegram    3h/12h  43/100  [CNN]       audioset
    mnetv2:     0.636   0.622   1.421   0.573       melspec     3h/3h   89/100  [CNN]       -
    psla:       0.646   0.627   1.364   0.580       melspec     1h/4h   11/100  [CNN]       imagenet
    effmnetv3:  0.611   0.587   1.472   0.594       melspec     4h/4h   46/50   [CNN]       audioset
    mnetv1:     0.623   0.604   1.488   0.617       melspec     3h/3h   46/50   [CNN]       -
    cnn14:      0.527   0.424   1.654   0.623       melspec     2h/4h   48/100  [CNN]       -
    cnn10:      0.561   0.455   1.584   0.637       melspec     1h/4h   16/75   [CNN]       -
    resnet38:   0.502   0.458   1.747   0.662       melspec     10/13h  77/100  [CNN]       -
    cnn6:       0.560   0.509   1.571   0.677       melspec     1h/2h   41/50   [CNN]       -
    hts:        0.613   0.594   1.564   0.693       melspec     5h/8h   37/50   [TR]        -
    passt:      0.616   0.610   1.571   0.699       melspec     10h/11h 46/50   [TR]        audioset
    ast:        0.535   0.480   1.672   0.702       chroma      30m/2h  7/20    [TR]        imagenet, audioset
    fnn:        0.535   0.496   1.694   0.744       mfcc        30m/1h  10/20   [DNN]       -
    resnet54:   0.576   0.487   1.746   0.816       melspec     3h/23h  14/100   [CNN]     -
    facrnn:     0.502   0.494   2.050   0.827       melspec     10h/11h 46/50   [CRNN]      -
    lstm:       0.425   0.362   2.007   0.906       melspec     1h/7h   4/30    [RNN]       -
    base:       -       -       -       1.034       -           -       -       -           -


## Trained on folder 1:

    fold 1: MAE between train and test:  1.495981003845884      
                CC      CCC     MSE     MAE       input         time    epochs  arch
    wvlmcnn14:  0.652   0.600   2.096   0.799     wavegram      13h     50
    psla:       0.563   0.481   2.700   0.935     melspec       11h     100     [CNN]
    passt:      0.635   0.586   2.287   0.960     melspec
    cnn10:      0.538   0.436   2.581   0.986     melspec
    hts:        0.461   0.383   2.892   1.148     melspec       10      50

## Trained on folder 2:

    fold2: MAE between train and test:  1.495981003845884
                CC      CCC     MSE     MAE       input         time    epochs  arch

    passt:      0.640   0.546   3.206   0.971     melspec
    wvlmcnn14:  0.585   0.521   3.544   0.972     wavegram      13h     50
    cnn10:      0.518   0.427   3.883   1.084     melspec       3h      50     [CNN]
    psla:       0.500   0.430   3.873   1.094     melspec               10
    hts:        0.399   0.349   4.710   1.339     melspec

## Trained on folder 3:

    fold3: MAE between train and test:  1.4915186923228314
                CC      CCC     MSE     MAE       input         time    epochs  arch
    wvlmcnn14:  0.713   0.643   2.755   0.796     wavegram      13h     50
    passt:      0.688   0.605   2.949   0.950     melspec
    cnn10:      0.555   0.411   4.078   1.042     melspec       3h      50      [CNN]
    psla:       0.414   0.338   5.130   1.188     melspec       2h      20      [CNN]
    hts:        0.520   0.361   4.231   1.270     melspec

## Trained on folder 4:

    fold4: MAE between train and test:  1.4915186923228314
                CC      CCC     MSE     MAE       input         time    epochs  arch
    wvlmcnn14:  0.651   0.571   3.251   0.922     wavegram      13h     50
    passt:      0.624   0.542   3.388   1.030     melspec
    cnn10:      0.569   0.418   3.898   1.089     melspec       3h      50      [CNN]
    psla:       0.519   0.445   4.041   1.136     melspec               50      [CNN]
    hts:        0.518   0.414   4.096   1.241     melspec

# Mean of all folders:

                CC      CCC     MSE     MAE      input         time    epochs  arch
    wvlmcnn14:  0.676   0.614   2.485   0.804    wavegram      13h     50      [CNN]
    passt:      0.638   0.578   2.680   0.922    melspec,128   10h     30      [TR] 
    cnn10:      0.548   0.429   3.205   0.968    melspec,64    3h      50      [CNN]
    psla:       0.495   0.433   3.723   1.041    melspec,128   6h      50      [CNN]
    hts:        0.502   0.420   3.499   1.138    melspec,128   11h     50      [TR]
    MAE-base:   -       -       -       1.380    -             -       -       -


