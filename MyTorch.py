import torch
import numpy as np
import time
import Dataset
import save_load_interface
import AdversarialAttackCleverHans



class Classifier:
    """
    This class provides an abstraction layer to PyTorch for training and testing models. 
    It also saves a training history of the test and validation accuracy. 
    ...

    Attributes
    ----------
    model (PyTorch model) : model that we want to use for the classifier 

    dataset_name (string) : name of the dataset that we want to use with the classifier

    device : "cpu" or "cuda" to train on GPU 

    nb_classes (int) : number of classes in the dataset 

    input_shape (int, int, int) : CxWxH

    optimizer (torch optim) : optimizer PyTorch object 

    criterion (torch.nn loss function) : loss function PyTorch object 

    Methods
    -------
    fit(train_loader, optimizer, criterion, nb_epoch)
        trains the PyTorch model with the training data from train_loader

    eval(test_loader, device)
        evaluates the model with the testing data from test_loader

    score() 

    predict_proba() 

    """
    
    def __init__(self, model, dataset_name, device, nb_classes, batch_size, input_shape, optimizer, criterion, seed):
        self.model = model 
        self.dataset_name = dataset_name
        self.device = device 
        self.model.to(device)
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.criterion = criterion
        
        # class attributes for training history 
        self.validation_accuracy = []
        self.train_accuracy = [] 
        self.train_loss = []
        self.validation_loss = []
        self.seed = seed 

        # load the datasets using Dataset.py 
        self.train_dataset, self.test_dataset = Dataset.getDataset(dataset_name)
        self.test_dataloader = Dataset.get_dataLoader(self.test_dataset, self.batch_size)
        # dont make train dataloader yet because we'll do it inside the fit function to create train and validation splits 


    def overwrite_datasets(self, train_dataset, test_dataset): 

        self.train_dataset = train_dataset 
        self.test_dataset = test_dataset
        self.test_dataloader = Dataset.get_dataLoader(self.test_dataset, self.batch_size)

    def fit(self, nb_epoch, train_split, save_iter, path='', adv_train=False):
        '''
        Train the model with dataset and compute validation and test score every epoch. 

                Parameters:
                        nb_epoch (int) : number of epoch for training 
                        train_split (double) : proportion of training dataset used for training the model 

                Returns:
                        void  
        '''
        # set up train and validation dataloaders with Dataset.py 
        train_loader, validation_loader = Dataset.train_valid_split(self.train_dataset, self.batch_size, train_split, seed=self.seed) 
        len_list = len(self.validation_loss)
        total_time = time.time()
        for i_epoch in range(nb_epoch): 
            running_loss = 0 
            correct_count = 0 
            all_count = 0
            time_0 = time.time() 
            self.model.train() 
            for i_batch, (images, labels) in enumerate(train_loader): 
                # transfer data on GPU here, cuz cannot transfer dataloader directly on GPU 
                images, labels = images.to(self.device), labels.to(self.device) 

                if adv_train:
                    self.model.eval()
                    images = AdversarialAttackCleverHans.PGD_augmentation(self.model, images, 0.031, (1/3) * 0.031, 10, np.inf)
                    self.model.train()

                # reset gradients 
                self.optimizer.zero_grad() 
                # compute outputs of model 
                predictions = self.model(images) 
                _, predictions_ = torch.max(predictions, 1)
                # compute correct predictions according to labels
                correct = (labels == predictions_).sum().item()
                correct_count += correct 
                all_count += labels.size(0)
                # compute loss for predictions 
                labels = labels.long()  # TODO: this is a quick patch, I have to fix this bug in AdversarialAttack.py 
                train_loss = self.criterion(predictions, labels) 
                # compute gradients 
                train_loss.backward() 
                # update weights according to gradients previously computed
                self.optimizer.step() 
                # update running loss 
                running_loss += train_loss.item() 


            # compute the accuracy on the testing dataset 
            acc_train = (correct_count / all_count) * 100 
            acc_validation, val_loss = self.score(validation_loader)   
            print("Epoch: {}, epoch time: {:.4f}, train loss: {:.4f}, val loss: {:.4f}, score train: {:.4f}, score val: {:.4f}".format(i_epoch+1 ,(time.time() - time_0)/60, running_loss / (i_batch+1), val_loss, acc_train, acc_validation))
            # if want to compute adversarial score every epoch
            # acc_validation, val_loss, adv_score = self.score_adversarial(validation_loader)   
            # print("Epoch: {}, epoch time: {:.4f}, train loss: {:.4f}, val loss: {:.4f}, score train: {:.4f}, score val: {:.4f}, adv. Score: {:.4f}".format(i_epoch+1 ,(time.time() - time_0)/60, running_loss / (i_batch+1), val_loss, acc_train, acc_validation, adv_score))
            
            # save checkpoint of model and metrics just in case smtg goes wrong 
            if (i_epoch+1) % save_iter == 0 : 
                fileName = path + 'epoch_'+ str(i_epoch+1+len_list) + '.pth'
                save_load_interface.save_checkpoint(self, fileName, is_ecoc=False)

            # save test, train and loss for this epoch
            self.train_accuracy.append(acc_train)
            self.validation_accuracy.append(acc_validation)
            self.validation_loss.append(val_loss)
            self.train_loss.append(train_loss.item())
        print("Training Time: {}".format((time.time()-total_time)/60))



    def predict_eval(self, dataLoader): 
        '''
        Evaluate the model previously trained on new data with the true labels

                Parameters:
                        dataLoader (DataLoader): new data 

                Returns:
                        X, y : list of predictions (X) and true value labels (y)
        '''
        self.model.eval() 
        X = np.array([])
        y = np.array([])
        with torch.no_grad(): 
            for images, labels in dataLoader: 
                # transfer data on GPU 
                images, labels = images.to(self.device), labels.to(self.device)
                # compute predictions of model 
                outputs = self.model(images)
                # compute index of highest value in prediciton outputs
                _, predictions = torch.max(outputs, 1)
                X = np.append(X, predictions.to("cpu").numpy())
                y =  np.append(y, labels.to("cpu").numpy())
        return X, y     
                     

    def score(self, dataLoader): 
        '''
        Compute the score (% accuracy) of the model on some data 

                Parameters:
                        dataLoader (DataLoader): data 

                Returns:
                        Percentage of accuracy of the classification with the model 
        '''
        self.model.eval()
        correct_count = 0 
        all_count = 0 
        running_loss = 0
        with torch.no_grad():
            for i_batch, (images, labels) in enumerate(dataLoader): 
                # transfer data on GPU 
                images, labels = images.to(self.device), labels.to(self.device)
                # compute predictions of model 
                outputs = self.model.forward(images)
                _, predictions = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels) 
                running_loss += loss.item() 
                # compute correct predictions according to labels
                correct = (labels == predictions).sum().item()
                correct_count += correct 
                all_count += labels.size(0) 
        return ((correct_count/all_count)*100), running_loss / (i_batch+1)

    def score_adversarial(self, dataLoader): 
        '''
        Compute the score (% accuracy) of the model on some data as well as the adversarial score for the same data
        '''
        self.model.eval()
        correct_count = 0
        correct_count_adv = 0
        all_count = 0 
        running_loss = 0
        for i_batch, (images, labels) in enumerate(dataLoader): 
            # transfer data on GPU 
            images, labels = images.to(self.device), labels.to(self.device)

            # compute predictions of model 
            outputs = self.model(images)
            _, predictions = torch.max(outputs, 1)
            loss = self.criterion(outputs, labels) 
            running_loss += loss.item() 

            # compute correct predictions according to labels
            correct = (labels == predictions).sum().item()
            correct_count += correct 
            all_count += labels.size(0) 

            # get indices of correctly classified images and get the count as well 
            ind_good = np.argwhere(predictions.to("cpu").detach() == labels.to("cpu"))
           
            # generate adversarial examples from validation images 
            adv_images = AdversarialAttackCleverHans.PGD_augmentation(self.model, images, 0.031, (2/3)*0.031, 10, torch.nn.CrossEntropyLoss(), np.inf)

            with torch.no_grad():
                outputs_adv = self.model(adv_images)
                _, predictions_adv = torch.max(outputs_adv, 1)
                correct_ = (labels[ind_good] == predictions_adv[ind_good]).sum().item()
                correct_count_adv += correct_ 
                
        return ((correct_count/all_count)*100), running_loss / (i_batch+1), ((correct_count_adv/correct_count)*100)


    # methods for training history 
    def get_train_accuracy(self): 
        '''
                Returns: list of training accuracy at each epoch saved during training  
        '''
        return self.train_accuracy

    def get_validation_accuracy(self): 
        '''
                Returns: list of train accuracy at each epoch saved during training  
        '''
        return self.validation_accuracy

    def get_training_loss(self): 
        '''
                Returns: list of training loss at each epoch saved during training  
        '''
        return self.train_loss

    def get_validation_loss(self): 
        '''
                Returns: list of validation loss at each epoch saved during training  
        '''
        return self.validation_loss

    def set_history(self, train_accuracy, validation_accuracy, train_loss): 
        '''
        Used when loading a checkpoint (a previously trained model that was saved on disk). Populates the train, test and loss at each epoch. 

                Parameters:
                        train_accuracy (list): list of training accuracy at each epoch during training 
                        validation_accuracy (list): list of train accuracy at each epoch during training
                        train_loss (list): list of training loss at each epoch during training 
                Returns:
                        void 
        '''
        
        self.train_accuracy = train_accuracy
        self.validation_accuracy = validation_accuracy 
        self.train_loss = train_loss
