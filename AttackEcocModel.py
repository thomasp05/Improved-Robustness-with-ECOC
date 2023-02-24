# main script for attacking ECOC ensembles 

import Models
import AdversarialAttackCleverHans
import torch
import save_load_interface
import numpy as np
import Metrics
import Dataset
from Loss_Function import cw_loss

# parameters for PyTorch model type and its training
dataset_name = "CIFAR10" #"Fashion-MNIST"  #Fashion-MNIST or CIFAR10 (must be written this way)
batch_size = 100  
seed = 42  # for making sure same images are used every time
filters = [32, 64, 128] # params A, B, C
filter2 = [16]            # param D
root = 'models_final/CIFAR10/AdvT/IndAdvt_2iter/indAdvt_2iter.pth' # for SIMPLE model, provide path of simple_1 model. For ENSEMBLE models, provide path of folder where all SIMPLE models are

# params for performing attacks 
attack_type = "FGSM"      # can be 'FGSM' or 'PDG' or 'CWL2' or 'auto' For CWL2, need to save the perturbations and run the script "cw_bound.py" to get the accuracy with a L2 bound. Otherwise, the perturbations generated in this script are not bounded. 
norm = np.inf             # norm of the attack (np.inf or 2)
max_iter = 100            # max iteration for PGD attack 
es = True                 # True: PGD^es, False: PGD
plus = False              # adaptation for removing operations that could cause gradient masking (i.e., softmax in soft voter of ensemble baseline models)
epsilon =  0.031          # bound the attack norm 
eps_step = 1/3 * epsilon  # epsilon step for PGD attack
loss = cw_loss            # torch.nn.CrossEntropyLoss()   # loss function for PGD attack 
nb_images = 2000            

# Parameters
seed = 42   # to make sure we use the same images everytime 
batch_size = 1 #100
nb_independent_models = 16
nb_bits_per_model = 1
save_adversarial_examples = False

# Params for loading model and saving attack images and metrics 
path_adversaries = "current_experimentation/PGD_0_031_tanh.pth"            # where the adversarial images will be saved                             
path_metrics = "checkpoints/cifar10/current_experimentation/metrics.npz"   # where the metrics (hamm dist, codes, etc) will be saved

# load W from files 
W = np.load('codewords/codeword_' + str(nb_bits_per_model*nb_independent_models) + '_bit.npy', allow_pickle=True) 
W[W==0] = -1   # scale codes to [-1, 1] range instead of [0,1] and get first code of list

# Model architecture details for ecoc models                                                             
model = Models.ecoc_ensemble_no_bn(W, nb_independent_models, filters, filter2, nb_independent_models*nb_bits_per_model, dataset_name) 

def main(): 
    print("\nGenerating Adversarial Attack" ) 
    # check if cuda available 
    print("CUDA available: {}\n".format(torch.cuda.is_available()))
    device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu") 

    # print experimentation details 
    print("Dataset: {}".format(dataset_name))
    print("Attack Type: {}, Epsilon: {}, Epsilon step: {}, Max Iter: {}, Nb Images: {}".format(attack_type, epsilon, eps_step, max_iter, nb_images))


    # get testing data
    _, test_dataset = Dataset.getDataset(dataset_name)
    test_dataloader = Dataset.get_dataLoader(test_dataset, batch_size)
 
    # Lad the ECOC model from files
    checkpoint = save_load_interface.load_checkpoint(root)
    model.load_state_dict(checkpoint['model'])
    full_model = torch.nn.Sequential(model, Models.ecoc_decoder(torch.from_numpy(W).to(device), no_tanh=plus))  # sequence of ecoc model and decoder to output the probabilities of each class 
    full_model.to(device)
    full_model.eval()

    ### Adversarial attack (generate adversarial examples)
    test_data = Dataset.get_test_subset(test_dataset, batch_size, nb_images, seed=seed)
    if attack_type == 'FGSM':
        data_perturbed = AdversarialAttackCleverHans.FGSM(full_model, batch_size, test_data, epsilon, norm)             
    elif attack_type == 'PGD':
        data_perturbed = AdversarialAttackCleverHans.PGD(full_model, batch_size, test_data, epsilon, eps_step, max_iter, norm, loss, early_stop=True)
    elif attack_type == 'CW':
        data_perturbed = AdversarialAttackCleverHans.CW_L2(full_model, batch_size, test_data, max_iter=max_iter)
    elif attack_type == 'auto': 
        data_perturbed = AdversarialAttackCleverHans.autoAttack(full_model, batch_size, test_data, epsilon, norm)


    # compute accuracy before and after the attack
    predictions_clean, true_labels_clean = Metrics.predict_eval(full_model, test_data, device) 
    predictions_perturbed, true_labels_perturbed = Metrics.predict_eval(full_model, data_perturbed, device)
    accuracy_clean = Metrics.score(predictions_clean, true_labels_clean) 
    accuracy_perturbed = Metrics.score(predictions_perturbed, true_labels_perturbed)    
    asr = Metrics.success_rate(predictions_clean, true_labels_clean, predictions_perturbed)
    print("Test accuracy before the attack: {}".format(accuracy_clean))
    print("Accuracy of classifier against attack: {}".format(accuracy_perturbed))
    print("Attack success rate: {}".format(asr))

  
    # Save adversarial examples
    if save_adversarial_examples: 
        save_load_interface.save_adversarial_examples(path_adversaries, data_perturbed)
    
        # save a bunch of metrics to file for later analysis
        labels_list, pred_labels_list, true_codes_list, inferred_codes_list, real_inferred_codes_list = Metrics.assess_codes(full_model, data_perturbed, torch.from_numpy(W).to(device), device) 
        labels_list = labels_list.numpy()
        pred_labels_list = pred_labels_list.numpy()
        true_codes_list = true_codes_list.numpy()
        inferred_codes_list = inferred_codes_list.numpy()
        real_inferred_codes_list = real_inferred_codes_list.numpy()
        np.savez(path_metrics, true_labels=labels_list, pred_labels_clean=predictions_clean, predicted_labels=pred_labels_list, true_codes=true_codes_list, predicted_codes=inferred_codes_list, real_predicted_codes=real_inferred_codes_list)



if __name__ == "__main__": 
    main() 
