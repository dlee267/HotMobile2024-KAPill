# Hotmobile2024-KAPill

# datasets

For each dataset:
- a `data` folder contains all the raw data collected from the sensor system
- a `events` folder contains all the events detected from the the `data` folder
- `fft` folder is the fft generated from the events found in the `events` folder

## KAA

The `KAA` folder contains the dataset using the different augmented bases:
- no augmentation
- rubber ball with spring
- metal ball with spring

The `NO_KAA` folder contains the dataset using the different bottles without augmented bases

# results.ipynb

All the code needed to reproduce the accuracy scores and features created from the events is found in `results.ipynb`.
It will generate the accuracy score graph `results3.pdf`.

Please run all the code cells in order to reproduce the results accurately.

# Special Thanks (Credit)

For svm tuning, we used this hyperparameter training model created by KevinHooah for AutoQual: task-oriented structural vibration sensing quality assessment leveraging co-located mobile sensing context found here!
