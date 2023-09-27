# KIRun speed estimation

Order of steps that need to be executed:

+ Create `.csv` file which has start/end times for all segments
+ Run `melspects.py` to extract features
+ Run `training.py` to train model

## Proposed File Structure:

<pre>
--kirun-speed-estimation  
  --data  
      --kirun-data-by-runner  
          -- RUNNER_X
              --RUNNER_X_RUN_Y  
          (...)  
      --melspec_data   
          --wdw5 (windowed files converted by melspects.py)
              --fold0
                  --0...000.npy
                  (...)
                  --x...xxx.npy
                  --features.csv
          (...)
      --metadata
          --original (unwindowed folds and gps-data i.e. gps.csv, meta.csv, steps.csv and all folds untouched)
          --wdw5 (windowed and ready for training)
              --fold0
                  --devel.csv
                  --fold0.csv
                  --test.csv
                  --train.csv
          (...)
      --results
          --wdw5
              --model_xyz (lowercase)
                  --Epoch_1
                  (...)
                  --Epoch_N
                  (...)
                  --test.yaml
  --venv (opt.)
  --datasets.py
  (...)
  --utils.py
</pre>