import numpy as np 
import torch.nn
import torch
import Models 
from Loss_Function import hinge_vec
import save_load_interface
from MyTorch_ecoc import Classifier_ecoc as Classifier_ecoc

# parameters for PyTorch model type and its training
dataset_name = "CIFAR10"
batch_size = 100
nb_epoch = 1000
lr = 0.0001 
seed = 42       
path_checkpoint = "ecoc_16_1/"  # were the checkpoints are saved
save_iter = 10                         # interval between checkpoints being saved
nb_independent_models = 16            # Number of shared feature extractors (for independent models, 16, 32 and 16 since they are independent)
nb_bits_per_model = 1
filters = [32, 64, 128]   # params A, B, C
filter2 = [16]            # param D
RegAdvt = False           # if want to perform RegAdvt

# Set to true to load a model and resume its training
load_model = False
load_model_filename = "models_final/CIFAR10/AdvT/IndAdvt_2iter/indAdvt_2iter.pth" # path of model       

# load W from files 
W = np.load('codewords/codeword_' + str(nb_bits_per_model*nb_independent_models) + '_bit.npy', allow_pickle=True) 
W[W==0] = -1   # codewords need to be [-1, 1] instead of [0, 1]

# Init ECOC Model
model = Models.ecoc_ensemble_no_bn(W, nb_independent_models, filters, filter2, nb_independent_models*nb_bits_per_model, dataset_name) 
# list of optimizers. One optimizer per ecoc ensemble member
optimizers = []
for model_ in model.models_ensemble: 
    optimizers.append(torch.optim.Adam(model_.parameters(), lr=lr))

criterion = hinge_vec() 

# Set dataset dependent params 
if dataset_name == "MNIST":
    input_shape = (1, 28, 28) 
    nb_classes = 10

elif dataset_name == "Fashion-MNIST": 
    input_shape = (1, 28, 28) 
    nb_classes = 10

elif dataset_name == "CIFAR10": 
    input_shape = (3, 32, 32)
    nb_classes = 10                             

print("\nTraining ECOC Model" ) 
print("Dataset: {}".format(dataset_name))
print("Batch size: {}, Nb of epochs: {}, learning rate: {}".format(batch_size, nb_epoch, lr))

#check if cuda is available. Training is much faster on the GPU. 
device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu") 

if load_model: 
    checkpoint = save_load_interface.load_checkpoint(load_model_filename)
    model.load_state_dict(checkpoint['model'])
    model.to(device)  # need to do it here otherwise the optimizer parameters are on cpu instead of device (or cuda if gpu available)
    classifier = Classifier_ecoc(model, W, dataset_name, device, nb_classes, batch_size, input_shape, optimizers, criterion, seed)
    classifier.set_history(checkpoint['train_accuracy'], checkpoint['validation_accuracy'], checkpoint['train_loss'], checkpoint['val_loss'], checkpoint['euclidean_dist'], checkpoint['hamming_dist'], checkpoint['hamming_dist_val'], checkpoint['euclidean_dist_val'])
else:      
    classifier = Classifier_ecoc(model, W, dataset_name, device, nb_classes, batch_size, input_shape, optimizers, criterion, seed) 

# Train ECOC model
classifier.fit_advt(nb_epoch, 0.98, save_iter, path=path_checkpoint)  # set full_data to false for testing. 1000 images are separated between the train and validation sets.  
