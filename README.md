# Hotmobile2024-KAPill

# Base Design

The 3d model base designs is found in the `pill_bottle_base_design` folder:
- the `Base.stl` is the base design we 3d printed to use for our research
- the `Augmented base.stl` is the proof of concept 3d design with a mock spring and ball mass to visualize our augmented base

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

For svm tuning, we used [this](https://github.com/KevinHooah/ML-model-hyperparameter-tuning) hyperparameter
training model created by `KevinHooah` for `AutoQual: task-oriented structural vibration sensing quality assessment leveraging co-located mobile sensing context`
[found here!](https://link.springer.com/article/10.1007/s42486-021-00073-3)
