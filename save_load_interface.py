from torch import save
from torch import load
import os   #use this import to check if the filename exists


def save_checkpoint(classifier, filePath, is_ecoc=False, is_indAdvt=False):
    '''
    save checkpoint for a given model. The checkpoint will persist the following information: model state_dict(), optimizer state_dict(), 
    validation accuracy list, train accuracy list, train loss list and criterion

            parameters : 
                        classifier (classifier object): classifier object of the model  
                        path (string): path where the checkpoint will be saved 
                        fileName (string): name of the file to be saved for the checkpoint 

    '''
    # check if file already exists and promp user for a new name if it does
    while os.path.isfile(filePath) == True:
        print("Choose another filename, this one already exists (include .pth")
        new_fileName = input("new file name:")
        filePath = new_fileName

    # create a checkpoint using a dictionary 
    if is_ecoc:   # the only difference is that for ecoc models, the lists for avg hamming and euclidean distance are saved in the checkpoint.
        checkpoint = { 
            'model': classifier.model.state_dict(), 
            'optimizer': classifier.optimizer.state_dict(), 
            'validation_accuracy': classifier.validation_accuracy, 
            'train_accuracy': classifier.train_accuracy, 
            'train_loss': classifier.train_loss, 
            'val_loss': classifier.validation_loss,
            'euclidean_dist':classifier.avg_euclidean_dist, 
            'hamming_dist': classifier.avg_hamming_dist,
            'hamming_dist_val':classifier.val_hamming_dist, 
            'euclidean_dist_val': classifier.val_euclidean_dist,
            'criterion': classifier.criterion,
            'seed':classifier.seed,
        }
    elif is_indAdvt and not is_ecoc: 
         checkpoint = { 
            'model': classifier.model.state_dict(), 
            'validation_accuracy': classifier.validation_accuracy, 
            'train_accuracy': classifier.train_accuracy, 
            'train_loss': classifier.train_loss, 
            'val_loss': classifier.validation_loss,
            'euclidean_dist':classifier.avg_euclidean_dist, 
            'hamming_dist': classifier.avg_hamming_dist,
            'hamming_dist_val':classifier.val_hamming_dist, 
            'euclidean_dist_val': classifier.val_euclidean_dist,
            'criterion': classifier.criterion,
            'seed':classifier.seed,
        }
    else: 
         checkpoint = { 
            'model': classifier.model.state_dict(), 
            'optimizer': classifier.optimizer.state_dict(), 
            'validation_accuracy': classifier.validation_accuracy, 
            'train_accuracy': classifier.train_accuracy, 
            'train_loss': classifier.train_loss, 
            'val_loss': classifier.validation_loss,
            'criterion': classifier.criterion,
            'seed':classifier.seed,
        }

    save(checkpoint, filePath)

def load_checkpoint(filePath): 
    '''
    Load a checkpoint that was previously saved using the save_checkpoint() method

            parameters : 
                        path (string): path where the checkpoint was saved
                        fileName (string): name of the file for the checkpoint 
            Returns : 
                    checkpoint 
    '''    
  
    if os.path.exists(filePath):
        # load the checkpoint corresponding to the fileName 
        checkpoint = load(filePath)
        return checkpoint
    else: 
        print("Error: the file name for the checkpoint doesn't exists") 
        return False


def save_adversarial_examples(filePath, dataLoader):
    '''
    save checkpoint for adversarial examples

            parameters :  
                        path (string): path where the checkpoint will be saved 
                        attackName (string): name of the attack 
                        fileName (string): name of the file for the checkpoint
    '''
    while os.path.isfile(filePath) == True: 
        print("Choose another filename, this one already exists (include .pth")
        new_fileName = input("new file name:")
        filePath = new_fileName

    # create a checkpoint for the dataLoader and all other important information 
    checkpoint = {
        'dataLoader': dataLoader, 
    }
    save(checkpoint, filePath) 

def load_adversarial_examples(filePath): 
    '''
    Load checkpoint of adversarial examples

            parameters :  
                        path (string): path where the checkpoint was saved 
                        attackName (string): name of the attack 
                        fileName (string): name of the file for the checkpoint
            Returns : 
                        checkpoint
    '''
    # check if the file exists 
    if os.path.isfile(filePath): 
        checkpoint = load(filePath) 
        return checkpoint
    else:  
        print("Error: the file name for the checkpoint doesn't exists")
        return False 

   
