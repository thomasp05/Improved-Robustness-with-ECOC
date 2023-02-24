# this script is for attacking the baseline models
import Models
import AdversarialAttackCleverHans
import torch
import save_load_interface
import numpy as np
import Metrics
import Dataset
from Loss_Function import cw_loss
import SimpleEnsemble

# parameters for PyTorch model type and its training
dataset_name = "Fashion-MNIST"  # MNIST, Fashion-MNIST or CIFAR10
batch_size = 100  
seed = 42  # for making sure same images are used every time
root = 'models_final/FMNIST/Baselines/' # for SIMPLE model, provide path of simple_1 model. For ENSEMBLE models, provide path of folder where all SIMPLE models are

# params for performing attacks 
attack_type = "FGSM"       # can be 'FGSM' or 'PDG' or 'CWL2' or 'auto
norm = np.inf          
max_iter = 100            # max iteration for PGD attack 
es = True                 # True: PGD^es, False: PGD
plus = False              # adaptation for removing operations that could cause gradient masking (i.e., softmax in soft voter of ensemble baseline models)
nb_images = 2000            
epsilon =  0.06          # bound the attack norm 
eps_step = 1/3 * epsilon  # epsilon step for PGD attack
loss = cw_loss            # torch.nn.CrossEntropyLoss()   # loss function for PGD attack 

## SIMPLE model                                                    
model = Models.simple([32, 32, 32], [4], dataset_name) # params are A, B, C and D for number of filters
checkpoint = save_load_interface.load_checkpoint(root)
model.load_state_dict(checkpoint['model'])

## ENSEMBLE_6 model 
# model = SimpleEnsemble.get_ensemble(6, root, dataset_name, no_softmax=plus)

## ENSEMBLE_16 model
# model = SimpleEnsemble.get_ensemble(16, root, dataset_name, no_softmax=plus)


# params for save_load interface
load_model_filename = "checkpoints/final_models/data/simple_models/simple_1/epoch_168.pth"                                            # file name of the model that is loaded if load_model is set to true

def main(): 
    print("\nGenerating Adversarial Attack" ) 
    # check if cuda available 
    print("CUDA available: {}\n".format(torch.cuda.is_available()))
    device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu") 
    model.to(device)

     # get testing data
    _, test_dataset = Dataset.getDataset(dataset_name)
    test_dataloader = Dataset.get_dataLoader(test_dataset, batch_size)


    # print experimentation details 
    print("Dataset: {}".format(dataset_name))
    print("Attack Type: {}, Epsilon: {}, Nb Images: {}".format(attack_type, epsilon, nb_images))

    ### Adversarial attack (generate adversarial examples)
    test_data = Dataset.get_test_subset(test_dataset, batch_size, nb_images, seed=seed)  
    if attack_type == 'FGSM':
        data_perturbed = AdversarialAttackCleverHans.FGSM(model, batch_size, test_data, epsilon, norm)              
    elif attack_type == 'PGD':
        data_perturbed = AdversarialAttackCleverHans.PGD(model, batch_size, test_data, epsilon, eps_step, max_iter, norm, loss=loss, Early_stop=es)
    elif attack_type == 'CWL2':
        data_perturbed = AdversarialAttackCleverHans.CW_L2(model, batch_size, test_data, max_iter)
    elif attack_type == 'auto': 
        data_perturbed = AdversarialAttackCleverHans.autoAttack(model, batch_size, test_data, epsilon, norm)


    # compute accuracy before and after the attack
    predictions_clean, true_labels_clean = Metrics.predict_eval(model, test_data, device) 
    predictions_perturbed, true_labels_perturbed = Metrics.predict_eval(model, data_perturbed, device)
    accuracy_clean = Metrics.score(predictions_clean, true_labels_clean) 
    accuracy_perturbed = Metrics.score(predictions_perturbed, true_labels_perturbed)    
    asr = Metrics.success_rate(predictions_clean, true_labels_clean, predictions_perturbed)
    print("Test accuracy before the attack: {}".format(accuracy_clean))
    print("Accuracy of classifier against attack: {}".format(accuracy_perturbed))
    print("Attack success rate: {}".format(asr))


if __name__ == "__main__": 
    main() 
