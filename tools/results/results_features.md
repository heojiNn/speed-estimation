
# Different features, same models:

Results on trying different input representations on the same architectures.  
Considered were FNN, CNN and Transformer (AST).  
FNN and CNN were self-implemented and not thouroughly tuned.  CNN based on CNN6.  
All trained on fold0.  

## Fold 0, WDW5:
    fnn ('dnn'):
        CC      CCC     MSE     MAE     input           epochs
        0.533   0.493   1.701   0.745   mfcc,7          20
        0.228   0.111   2.153   0.814   waveform        20
        0.389   0.385   2.489   0.851   melspec,m128    20
        0.284   0.24x   2.335   0.921   melspec,m64     20
        0.533   0.509   2.176   0.949   mfcc,13         20

    cnn ('specnn'):
        CC      CCC     MSE     MAE     input           epochs
        0.548   0.510   1.704   0.770   chromagram,12   10       
        0.242   0.150   2.209   0.819   melspec,64      10          
        0.207   0.153   2.301   0.903   melspec,128     10          
        0.426   0.423   2.892   1.200   mfcc,12         10
                    
    ast: frequency stride=10, time stride=10
        CC      CCC     MSE     MAE     input           epochs  pretrain    patches
        0.535   0.480   1.672   0.702   chromagram,16   20      imnet       49
        0.368   0.292   2.077   0.824   melspec,64      20      imnet       245
        0.436   0.405   1.980   0.858   melspec,128     20      imnet       588


## MEL-SPECTROGRAM ADJUSTMENTS:

Adjusting stft/fft settings wasn't further investigated as the default yielded the best results on cnn10.  
Default: sr: 16kHz, window=nfft=320, hoplength=160, minF=50(cut low noise)Hz, maxF=sr//2=8000Hz.  

### N_MELS == 64:  

    mae                 CC      hparam                   architecture   [arch]      epoch      improvement     time

    0.639259397983551   0.552   Adam_b32_1e-3            cnn10_fa       [cnn]       16/50      38.1821 %       3h                                
    0.6935730576515198  0.614   AdamW_b32_1e-4           hts_fa         [tr]        37/100     32.9299 %       16h
    0.7741907238960266  0.544   Adam_b32_1e-3            facrnn         [crnn]      37/50      25.1339 %       11h
    0.8408348560333252          Adam_b32_1e-3            cnn14          [cnn]       10/10      18.6893 %
    0.893455445766449           Adam_b16_1e-3            lstm           [rnn]       7/10       13.6008 %
    0.9110188484191895  0.403   Adam_b32_1e-4            ast            [tr]        18/20      11.9023 %       11h
    0.9211254715919495          Adam_b16_1e-3            dnn            [dnn]       2/10       10.9250 % 

    1.0341010816006755  BASELINE

### N_MELS == 128:
    0.63068026304245    0.589                            pslr:          [cnn]       6/10       39.0117 %
    0.6414166688919067  0.595                            cnn10m128      [cnn]       39/60      37.9735 %
    0.6993042230606079  0.616                            passt          [tr]        30/50      34.1729 %       10h
    0.8090290427207947  0.378   Adam_b16_1e-3            dnn            [dnn]       9/50       23.8446 %       12min/82 