import sys 
import Models
from MyTorch import Classifier
import torch 
from matplotlib import pyplot as plt 
import save_load_interface
import Metrics


# parameters for PyTorch model type and its training
dataset_name = "Fashion-MNIST" #"CIFAR10"  
batch_size = 100
nb_epoch = 400
lr = 0.001       
seed = 42
save_iter = 100
adv_training = False
path_checkpoint = 'checkpoints/cifar10/current_experimentation/'   #where checkpoints will be saved

model = Models.simple([32, 32, 32], [4], dataset_name) # params are A, B, C and D for number of filters
                                   

load_model = False # if want to load a model and resume its training
load_model_filename = ""     

def main(): 
    print("\nRobustness to adversarial attacks" ) 
    # check if cuda available 
    print("CUDA available: {}\n".format(torch.cuda.is_available()))
    device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu") 

    # print experimentation details 
    print("Dataset: {}".format(dataset_name))
    print("Batch size: {}, Nb of epochs: {}, learning rate: {}".format(batch_size, nb_epoch, lr))

    ## set dataset size 
    if dataset_name == "Fashion-MNIST": 
        input_shape = (1, 28, 28) 
        nb_classes = 10

    elif dataset_name == "CIFAR10" or dataset_name == "SVHN": 
        input_shape = (3, 32, 32)
        nb_classes = 10
    else: 
        sys.exit("\nERROR: Invalid dataset_name")


    if load_model: 
        checkpoint = save_load_interface.load_checkpoint(load_model_filename)
        model.load_state_dict(checkpoint['model'])
        model.to(device)  # need to do it here otherwise the optimizer parameters are on cpu instead of device (or cuda if gpu available)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        criterion = checkpoint['criterion']
        classifier = Classifier(model, dataset_name, device, nb_classes, batch_size, input_shape, optimizer, criterion, seed) 
        classifier.set_history(checkpoint['train_accuracy'], checkpoint['validation_accuracy'], checkpoint['train_loss'])

    else:      
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        classifier = Classifier(model, dataset_name, device, nb_classes, batch_size, input_shape, optimizer, criterion, seed) 

    
    ### Train PyTorch model (if model is not pretrained)
    classifier.fit(nb_epoch, 0.98, save_iter, path=path_checkpoint, adv_train=adv_training)    

    ### Compute metrics (accuracy, etc) and generate figures  
    X, y = classifier.predict_eval(classifier.test_dataloader)  
    accuracy = Metrics.score(X,y )
    print("Test accuracy: {}".format(accuracy))
    plt.plot(classifier.get_validation_accuracy(), label="Validation accuracy")
    plt.plot(classifier.get_train_accuracy(), label="Train accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__": 
    main() 
