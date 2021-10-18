# Modulation Pattern Detection Using Complex Convolutions in Deep Learning
## About
This repository contains code to reproduce experiments for, "Modulation Pattern Detection Using Complex Convolutions in Deep Learning" which was published in the 25th International Conference on Pattern Recognition (ICPR2020). 

## Data
Data for this submission (RML2016.10a.tar.bz2) can be found at: https://www.deepsig.io/datasets. To ensure proper execution of the code, be sure the data is saved as 'RML2016.10a_dict.pkl' or 'RML2016.10a_dict.dat'.

## Code

The following code will execute an example experiment training and testing across all SNR levels: (be sure to include the path to the dataset)
```
python3 run.py --data_directory path_to_data --train_SNRs -20 18 --test_SNRs -20 18 
```
The code automatically saves and stores results into folders in the local directory. 

*Disclaimer: All code is written in Keras
