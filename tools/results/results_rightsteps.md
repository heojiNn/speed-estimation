
# This markdown contain the experimental results of training on estimating the number of *right* steps.

## <span style="color:Orange"> Baseline </span>
### mae([df_train['steps'].mean()] * len(df_test), df_test['steps']:

wdw-size 5:

    fold0 0.94970195025004
    fold1 1.4993552693187906
    fold2 1.314672627164834
    fold3 1.2276695984401191
    fold4 0.9694819596915019

wdw-size 10

    fold0 2.089984461779507
    fold1 2.0949227269644517
    fold2 1.9667031252505414
    fold3 2.625360782550664
    fold4 2.0971881791345846

wdw-size 20

    fold0 3.2454338799355584
    fold1 3.8942379220558574
    fold2 3.8835774406489056
    fold3 3.671442614909922
    fold4 3.754644091342293
 
### Experimentation on fold0, wdw5 - RIGHT STEPS:  
MAE baseline: <span style="color:red">0.9497019502500</span>

    mae                 hparam                   architecture        epoch      improvement

    0.8511725068092346  default                  dnn                 3/10       10.3748 %
    0.8520665168762207  Adam_16_1e-3             cdnn                8/10       10.2806 %
    0.8541439771652222  default                  cnnlstm1            10
    0.8572724461555481  default                  cnn10lstm           10
    0.8611426949501038  default                  lstm                1
    0.8683491349220276  AdamW_b32_1e-4           hts_fa              3/10       8.5661 %
    0.875189483165741   Adam                     fdycrnn             10          
    0.8766012787818909  default                  cnnlstm2            5
    0.8811988830566406  AdamW_b32_1e-4           hts_sa              4/10       7.2131 %
    0.8830069303512573  Adam_b32_5e-5            ast                 10/10   
    0.8850244283676147  RMSprop                  cnn10               1 
    0.9036572575569153  default                  facrnn              5
    0.9116702079772949  default                  passt               33/100     4.0046 %

    < MAE-BASELINE

    1.0357235670089722  default                  wvlmcnn14           3/20
    1.1213467121124268  default                  resnet38            25
    1.1858084201812744  b32                      cnn14               10
    1.5508363246917725  default                  resnet22            10
    1.5939699411392212  default                  resnet54            10   


------------------------------------------------------------------------------------------------------------------------
# <span style="color:Orange"> Training </span>
### wdw-size 5:  
fold0:  baseline: 0.9497019502500

    mae                 hparam                          architecture        epochs

    0.8511725068092346  default                         dnn                 3/10            10.3748 %
    0.8541439771652222  default                         cnnlstm1            10
    0.8572724461555481  default                         cnn10lstm           10
    0.8584260940551758  default                         gru                 7/10
    0.8604676723480225  SGD_16_0001                     cnnlstmv1           5
    0.8715502023696899  AdamW_16_0001                   lstm                9/10
    0.8766012787818909  SGD_16_0001                     cnnlstmv2           5
    0.8850244283676147  RMSprop_b16_lr0001              cnn10               1
    0.8852840662002563  RMSprop_b8_lr0001               cnn10               1
    0.8859564065933228  RMSprop_b12_lr0001              cnn10               1
    0.9004689455032349  SGD_16+bn                       cnnlstmv1           10
    0.9200317859649658  SGD_b16_lr0001                  cnnlstmv1           10
    0.9268344640731812  SGD_b16_lr0001                  cnn10lstm           5
    0.9327170252799988  RMSprop_b16_lr000125            cnn10               1
    0.9490311741828918  SGD_b32_lr0001                  cnnlstm_v1          10
    < bl
    1.0294667482376099  SGD_b16_lr0001                  cnnlstmv1           50
    1.0357235670089722  default                         wvlmcnn14           3/20
    1.0435739755630493  RMSprop_b16_lr00005             cnn10               5
    1.0462297201156616  RMSprop_b14_lr0001              cnn10               1
    1.0472723245620728  RMSprop_b16_lr0001              cnn10               5  
    1.0551643371582031  RMSprop_b16_lr0001              cnn10lstm           5
    1.06451416015625    RMSprop_b16_lr0001              cnn10               10
    1.0695089101791382  RMSprop_b16_lr0001              cnn10               200
    1.072891354560852   SGD_b16_lr0001                  cnnlstm_v1          1
    1.082970142364502   RMSprop_b64_lr0001              cnn10               25
    1.1213467121124268  SGD_b16_lr0001                  resnet38            25
    1.1458693742752075  Adam_b32_lr0001                 cnn10               100
    1.1459215879440308  Adam_b16_lr0001                 cnn10               200
    1.154768466949463   SGD_b16_lr0001                  resnet38            10
    1.1644337177276611  SGD_b16_0001                    cnn10               25
    1.1703119277954102  Adam_16_0001                    cnn10               25
    1.1858084201812744  SGD_b32_lr0001                  cnn14               10
    1.1873067617416382  RMSprop_b32_lr_0001             cnn10               25
    1.1896227598190308  SGD_b16_lr0001                  cnn14               100
    1.2045409679412842  SGD_32_0001                     resnet38            10
    1.2248765230178833  SGD_16_001                      resnet38            20
    1.2427740097045898  RMSprop_b16_lr0001pl            cnn10               10
    1.2459713220596313  SGD_b16_lr0001                  cnn10               100
    1.2467974424362183  RMSprop_b16_lr0001              cnn10               25
    1.248983383178711   RMSprop_b16                     cnn10               50
    1.255164384841919   SGD_b16_lr001                   resnet38            10
    1.2991050481796265  SGD_b32_lr0001                  cnn10               50
    1.344502329826355   SGD_b16_lr0001                  cnn14               25
    1.3725706338882446  Adam_b16_lr0001                 ast                 5
    1.5508363246917725  SGD_b16_lr0001                  resnet22            10
    1.5939699411392212  SGD_b16_lr0001                  resnet54            10

fold1: baseline: 1.4993552693187906

    mae                 foldername                      architecture        epochs

    1.4733175039291382  SGD_b16                         cnnlstmv1           5
    1.487230658531189   f1_Adam_b16_lr0001              cnn10               25
    1.5000678300857544  f1_RMSprop_b32_lr0001           cnn10               100
    1.5498558282852173  f1_RMSpropb16_lr0001            cnn10               25
    1.5882947444915771  f1_SGD_b16_lr0001               cnn10               100
    1.5968106985092163  f1_SGD_b16_lr0001               cnn14               100
    1.6558701992034912  f1_SGD_b16_lr0001               cnn10               25

fold2: baseline: 1.314672627164834

    mae                 hparam                          architecture        epochs

    1.2218313217163086  SGD_16                          cnnlstmv1           5
    1.2958592176437378  RMSprop_b32_lr0001              cnn10               10
    1.308960199356079   Adam_b16                        cnn10               25
    < bl
    1.401856541633606   SGD_b32_lr0001                  cnn10               10           

fold3: baseline: 1.2276695984401191

    mae                 hparam                          architecture        epochs

    1.1512519121170044  SGD_b16_lr0001                  cnnlstmv1
    < bl
    1.2807873487472534  RMSprop_b16_lr0001              cnn10               10

fold4: baseline: 0.9694819596915019

    mae                 hparam                          architecture        epochs

    0.8698278665542603  SGD_b16                         cnnlstmv1
    0.8906255960464478  RMSprop_b16_lr0001              cnn10               50
    < bl
    1.216536521911621   RMSprop_b16_lr0001              cnn10               1


------------------------------------------------------------------------------------------------------------------------

TODO:
    Recreate

### wdw-size 10:  
fold0: baseline: 2.089984461779507

    mae                  hparam                          architecture        epochs      improvement

    1.9725992679595947   SGD_b16_lr0001                 lstm                5/10         5.6166 %
    2.0035433769226074   AdamW_b12_1e-4_nmel128         ast                 4/10         4.1360 %            158 min
    2.028965473175049    AdamW_b8_5e-5_nmel128          ast                 4/10         2.9196 %            177 min
    2.057274103164673    RMSprop_b16_lr0001             cnn10               10/25
    2.1634371280670166   SGD_b32_0.001                  cnn14               15/20        -3.5145 %


invalid:
    1.981059193611145                                   ast
    2.0828325748443604  RMSprop_b16_lr0001              cnn10               25



------------------------------------------------------------------------------------------------------------------------

### wdw-size 20:  
fold0: baseline: 3.2454338799355584

    mae                 hparam                          architecture        epochs


invalid:
    3.2594223022460938   SGD_b16_lr0001                  lstm               20/25      -0.4310 %

## Applying best model to variations in mel-spec transformations:
lstm,SGD_b16
f_max   /   n_mel   

    mae                 epochs      f_max       n_mel
    0.8511725068092346  5           256         64                              10.3748 %
    0.8514090180397034  5           128         64
    0.8514558672904968  5           1024        64
    0.8515311479568481  5           4096        64
    0.8697386384010315  5           256         128


## Applying best model to different audio features:

### mfcc:
lstm
n_mfcc = 13 : number of cepstral coefficients obtained after applying DCT on the log-Mel-spectrogram

    mae                     feature         epochs      hparams
    0.8523368835449219      mfcc            8/20        n_mfcc:7        10.2522 %
    0.8587058186531067      mfcc            6/20        n_mfcc:13       9.5815 %
    0.8589573502540588      mfcc            13/20       n_mfcc:4        9.5551 %
    0.8590814471244812      mfcc            14/20       n_mfcc:10       9.5420 %
    0.8625020980834961      mfcc            9/10        n_mfcc:20       9.1818 %
    0.8775721788406372      mfcc            10/10       n_mfcc:40  

### chroma_stft:

    mae                     feature         epochs      hparams
    0.8548282980918884      chroma_stft     6/20        n_chroma:12     9.9898 %
    0.8549916744232178      chroma_stft     4/10        n_chroma:6      9.9726 %

### chroma_cqt:

    mae                     feature         epochs      hparams
    0.8520199656486511      chroma_cqt      17/20       n_chroma:12     10.2855 %
