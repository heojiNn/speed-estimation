# KIRun speed estimation

Results after Cross-Validation:

            model       | CC      CCC     MSE     MAE      input         time    epochs  arch
            wvlmcnn14:  | 0.676   0.614   2.485   0.804    wavegram      13h     50      [CNN]
            passt:      | 0.638   0.578   2.680   0.922    melspec,128   10h     30      [TR] 
            cnn10:      | 0.548   0.429   3.205   0.968    melspec,64    3h      50      [CNN]
            psla:       | 0.495   0.433   3.723   1.041    melspec,128   6h      50      [CNN]
            hts:        | 0.502   0.420   3.499   1.138    melspec,128   11h     50      [TR]
            MAE-base:   | -       -       -       1.380    -             -       -       -

- Target outputs/labels are all steps in windows of 5 seconds (best performing setting)
- Inputs are melspectrograms (settings according to code)
