"""
# Author
Jakob Krzyston (jakobk@gatech.edu)

# Last Updated
June 10, 2020

# Purpose
Enable rapid, organized experimentation of algorithms for I/Q data processing

"""
## Import packages and functions
import os, argparse
import numpy as np
import load_data, build_models, train_models
import overall_acc_and_conf_matrix, snr_acc_and_conf_matrix
import snr_plots, activation_maximizations


## Handle input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=False, default = 'RML2016')
parser.add_argument('--data_directory', type=str, required= True)
parser.add_argument('--samples', type=int, required=False, default=1000)
parser.add_argument('--train_pct', type=int, required=False, default = 50)
parser.add_argument('--train_SNRs', type=int, nargs='+', required=True)
parser.add_argument('--test_SNRs', type=int, nargs='+', required=True)
parser.add_argument('--load_weights', type=str, required=False, default = 0)
parser.add_argument('--trial', type=int, required=False, default = 0)
args = parser.parse_args()

## Experimental parameters
# Extract the name of the dataset
dataset = args.dataset

# Determine the train/test split
train_pct = args.train_pct/100

# Number of samples from each class per dB SNR to be used
samples = args.samples

# Specify SNRs to conduct the experiment
train_SNR_l = args.train_SNRs[0]
train_SNR_h = args.train_SNRs[1]+2
train_SNR_step = 2

test_SNR_l = args.test_SNRs[0]
test_SNR_h = args.test_SNRs[1]+2
test_SNR_step = 2

# Determine
if train_SNR_l == test_SNR_l and train_SNR_h == test_SNR_h and train_SNR_step == test_SNR_step:
    train_test_SNR_eq = True
    print('The training and testing SNRs are the same')
else:
    train_test_SNR_eq = False
    print('The training and testing SNRs are not the same')

# If already trained, save time and load the saved weights
load_weights = args.load_weights

# Specify file tag to ID the results from this run
# If splitting the data by SNR, include the SNR bounds that were used to train the network in the tag. 
# The bounds should not overlap as this would defeat the purpose of the experiment
tag = dataset+'_train_'+str(train_SNR_l)+'_'+str(train_SNR_h)+'_test_'+str(test_SNR_l)+'_'+str(test_SNR_h)+'_trial_'+str(args.trial)

# Setup directories to organize results 
sub_folders = ['Weights', 'Figures', 'Computed_Values']
for i in range(len(sub_folders)):
    path = os.path.join(os.getcwd(),tag+'/'+sub_folders[i])
    os.makedirs(path, exist_ok = True)


## Load data
Imported_data = load_data.load(dataset,args.data_directory,samples,train_pct,train_SNR_h,train_SNR_l,train_SNR_step,test_SNR_h,test_SNR_l,test_SNR_step,train_test_SNR_eq) 


## Build the architectures
Models, model_names = build_models.build(Imported_data['sample_sz'], Imported_data['classes'], dr = 0.5)


## Train/ load weights for the built models
# If training, will plot and save the training & validation loss curves in the 'Figures' folder
for i in range(len(model_names)):
    model = Models[i]
    model_name = model_names[i]
    train_models.train(model, model_name, Imported_data['X_train1'], Imported_data['X_test1'], Imported_data['Y_train1'], Imported_data['Y_test1'], tag, load_weights, epochs=100, batch_size = Imported_data['batch_size'])
    
    
## Verify which dataset to use when testing from here on, should it belong to a different distribution than the training data
if train_test_SNR_eq == True:
	X_test = Imported_data['X_test1']
	Y_test = Imported_data['Y_test1']
else:
	X_test = Imported_data['X_test2']
	Y_test = Imported_data['Y_test2']
    
    
## Overall accuracy and confusion matrix for the corresponding models
for i in range(len(model_names)):
    model = Models[i]
    model_name = model_names[i]
    overall_acc_and_conf_matrix.eval(model, model_name, X_test, Y_test, tag, Imported_data['classes'], batch_size = Imported_data['batch_size'])
    

## Accuracy and confusion matrix for the corresponding models at each SNR
for i in range(len(model_names)):
    model = Models[i]
    model_name = model_names[i]
    snr_acc_and_conf_matrix.eval(Imported_data['test_snrs'], model, model_name, Imported_data['test_samples'], Imported_data['test_labels'], tag, Imported_data['classes'], batch_size = Imported_data['batch_size'])
    

## Compute the classification accuracy by SNR for each model
# Extract the accuracy by SNR for each model
snr_accs = np.zeros((len(model_names), len(Imported_data['test_snrs'])))
for i in range(len(model_names)):
    model_name = model_names[i]    
    acc_name = os.getcwd() + '/' + tag + '/Computed_Values/' + model_name + '_SNR_Accuracy.npy'
    # Extra elements are added to the array in the saving process (not sure why)
    # This will remove those elements and include the correct values 
    snr_accs[i,:] = np.fromfile(acc_name, dtype = 'float32')[-int(len(Imported_data['test_snrs'])):]
# Compute SNR accuracies
snr_plots.plot(Imported_data['test_snrs'], snr_accs, model_names, tag)


# Compute and save the activation maximizations
activation_maximizations.compute(model_names, Models, Imported_data['classes'], tag, Imported_data['mods'], Imported_data['data_max'], Imported_data['data_min'], Imported_data['sample_sz'])