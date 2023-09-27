import pandas as pd

steps = pd.read_csv(r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\metadata\original\gps.csv')
gps = pd.read_csv(r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\metadata\original\steps.csv')

for index, row in gps.iterrows():
    gps.loc[index, 'steps_5sec'] = len(steps.loc[(
        (steps['file'] == row['file'])
        & (steps['elapsed'] >= row['elapsed'] - 2.5)
        & (steps['elapsed'] <= row['elapsed'] + 2.5)
    )])
print(gps.corr().loc['steps_5sec', 'speedGPS'])
# 0.165933
# AT: correlation is very low probably because GPS sensor is very noisy
