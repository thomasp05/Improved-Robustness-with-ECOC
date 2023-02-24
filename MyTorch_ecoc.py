import torch.nn.functional as F 
import torch
import numpy as np
import time
import Dataset
import save_load_interface
import Models
import AdversarialAttackCleverHans


class Classifier_ecoc:
    """
    This class provides an abstraction layer to PyTorch for training and testing models. 
    It also saves a training history of the test and validation accuracy. 
    ...

    Attributes
    ----------
    model (PyTorch model) : model that we want to use for the classifier 

    W : Codeword matrix for ECOC codes

    dataset_name (string) : name of the dataset that we want to use with the classifier

    device : "cpu" or "cuda" to train on GPU 

    nb_classes (int) : number of classes in the dataset 

    input_shape (int, int, int) : CxWxH

    optimizer (torch optim) : optimizer PyTorch object 

    criterion (torch.nn loss function) : loss function PyTorch object 

    seed : For np.random used for train/validation split of the training data

    """
    
    def __init__(self, model, W, dataset_name, device, nb_classes, batch_size, input_shape, optimizer, criterion, seed):
        self.model = model 
        self.dataset_name = dataset_name
        self.device = device 
        self.model.to(device)
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.criterion = criterion
        self.W = torch.from_numpy(W).to(self.device) #this is the matrix of codewords
        self.seed = seed # fix seed for train/validation split when I want to resume a previous train in order to get the same train/validation split. 

        # class attributes for training history 
        self.validation_accuracy = []
        self.train_accuracy = []      
        self.train_loss = []
        self.validation_loss = []

        self.avg_hamming_dist = [] 
        self.avg_euclidean_dist = []
        self.val_hamming_dist = []
        self.val_euclidean_dist = [] 

        # load the datasets using Dataset.py 
        self.train_dataset, self.test_dataset = Dataset.getDataset(dataset_name)
        self.test_dataloader = Dataset.get_dataLoader(self.test_dataset, self.batch_size)
        self.train_dataloader = Dataset.get_dataLoader(self.train_dataset, self.batch_size)
        # dont make train dataloader yet because we'll do it inside the fit function to create train and validation splits 

    def fit(self, nb_epoch, train_split, save_iter, advt=False, path=''):
        '''
        Train the model with dataset and compute validation and test score every epoch. 

                Parameters:
                        nb_epoch (int) : number of epoch for training 
                        train_split (double) : proportion of training dataset used for training the model 
                        save_iter (int) : interval at which model is saved during training
                        path (string) : path where the model is saved 
                        adv_train (bool) : Wheter adversarial training is used or not

                Returns:
                        void  
        '''
        # set up train and validation dataloaders with Dataset.py 
        train_loader, validation_loader = Dataset.train_valid_split(self.train_dataset, self.batch_size, train_split, seed=self.seed)
   
        total_time = time.time()
        len_list = len(self.validation_loss)
        for i_epoch in range(nb_epoch): 
            running_loss = 0 
            mean_hamming = 0
            mean_euclidian = 0
            time_0 = time.time() 
            self.model.train() 
            for i_batch, (images, labels) in enumerate(train_loader): 

                # transfer data on GPU here, cuz cannot transfer dataloader directly on GPU 
                images, labels = images.to(self.device), labels.to(self.device)  

                # if RegAdvt, compute PGD for images as described in Madry and Al 
                if advt: 
                    decoder = Models.ecoc_decoder(self.W, no_tanh=False)
                    full_model = torch.nn.Sequential(self.model.eval(), decoder.eval()) 
                    images = AdversarialAttackCleverHans.PGD_augmentation(full_model, images, 0.031, (1/3) * 0.031, 10, np.inf) # the parameters for PGD can be changed here.
                    self.model.train()
               
                # reset gradient
                self.optimizer.zero_grad()

                # compute inferred code of model 
                code = self.model(images)   
               
                # now, we need to get the code from W for the classes in labels 
                real_codes = self.W[labels,:]

                # new optimized hinge loss 
                avg_loss = self.criterion(code, real_codes) 
                
                avg_loss.backward()  # I think this is equivalent to calling backward on each loss individually.
                self.optimizer.step() 
                running_loss += avg_loss.item()

                # compute some metrics that will be saved at each checkpoint
                with torch.no_grad(): 
                    code = torch.tanh(code)    # apply activation function
                    # compute mean euclidian distance between real and inferred codes
                    mean_euclidian += (code - real_codes).pow(2).sum(1).sqrt().mean().to("cpu").numpy()  # compute euclidean distance between predicted and true codewords
                    
                    # compute the hamming distance between predicted and true codewords
                    code[code>0] = 1
                    code[code<=0] = -1
                    hamming_dist = F.relu( (code * real_codes) * (-1) ).sum(1)
                    mean_hamming +=  hamming_dist.mean().to("cpu").numpy()

            # save avg euclidiean and hamming distance for displaying at the end of training 
            avg_hamming = mean_hamming / (i_batch+1)
            avg_eucl = mean_euclidian / (i_batch+1)
            self.avg_hamming_dist.append(avg_hamming)
            self.avg_euclidean_dist.append(avg_eucl)

            # now we need to compute the hamming and euclidean distance for the validation set 
            hamming_dist_validation, euclidean_dist_validation, val_loss = self.ecoc_eval_metrics(validation_loader)  # compute metrics with validation images
            self.validation_loss.append(val_loss)
            self.val_hamming_dist.append(hamming_dist_validation)
            self.val_euclidean_dist.append(euclidean_dist_validation)
            self.train_loss.append(running_loss / (i_batch+1))

            # print epoch metrics 
            print("Epoch: {}, epoch time: {:.4f}, train loss: {:.4f}, val loss: {:.4f}, train ham: {:.4f}, train eucl: {:.4f}, val ham: {:.4f}, val eucl: {:.4f} ".format(i_epoch+1+len_list ,(time.time() - time_0)/60, running_loss / (i_batch+1), val_loss, avg_hamming, avg_eucl, hamming_dist_validation, euclidean_dist_validation)) 

            # save checkpoint of model and metrics just in case smtg goes wrong 
            if (i_epoch+1) % save_iter == 0 : 
                fileName = path +'epoch_'+ str(i_epoch+1+len_list) + '.pth'
                save_load_interface.save_checkpoint(self, fileName, is_ecoc=True)
        print("Total training Time: {}".format((time.time()-total_time)/60))


    def fit_advt(self, nb_epoch, train_split, save_iter, path=''):
        '''
        Train ECOC ensembles with IndAdvt (adversarial perturbations generated for each binary classifier) 
                Parameters:
                        nb_epoch (int) : number of epoch for training 
                        train_split (double) : proportion of training dataset used for training the model 
                        save_iter (int) : interval at which model is saved during training
                        path (string) : path where the model is saved 
                Returns:
                        void  
        '''
        # set up train and validation dataloaders with Dataset.py 
        train_loader, validation_loader = Dataset.train_valid_split(self.train_dataset, self.batch_size, train_split, seed=self.seed)
   
        total_time = time.time()
        len_list = len(self.validation_loss)
        for i_epoch in range(nb_epoch): 
            running_loss = 0 
            mean_hamming = 0
            mean_euclidian = 0
            time_0 = time.time() 
            self.model.train() 
            for i_batch, (images, labels) in enumerate(train_loader): 

                # transfer data on GPU here, cuz cannot transfer dataloader directly on GPU 
                images, labels = images.to(self.device), labels.to(self.device)    

                code =  torch.empty([0]).to(self.device) 
                count = 0
                tot_loss = 0
                for model_, optim in zip(self.model.models_ensemble, self.optimizer): 
                       
                        # if adversarial training, compute PGD for images as described in Madry and Al 
                        # Need a decoder for computing the adversarial examples with PGD
                        train_img = AdversarialAttackCleverHans.PGD_binary_augmentation(model_, images, 0.031, (1/3) * 0.031, 10, np.inf) # for now the loss function is useless. I keep it in case I want to modify the cleverhans code to be able to change it.
                        
                        model_.train()
                        optim.zero_grad()
                        code_ = model_(train_img)
                        code = torch.cat([code, code_], dim=1) 

                        # now, we need to get the code from W for the classes in labels 
                        real_codes = self.W[labels,:]
                        real_codes = real_codes[:, count]

                        # new optimized hinge loss 
                        avg_loss = self.criterion(code_, real_codes.unsqueeze(1)) 
                        avg_loss.backward()  # I think this is equivalent to calling backward on each loss individually.
                        optim.step() 
                        tot_loss +=  avg_loss.item()
                        count += 1
                
                running_loss += tot_loss  
                real_codes = self.W[labels,:]

                # appliquer la fonction d'activation en sortie pour scale les codes
                with torch.no_grad(): 
                    code = torch.tanh(code) 

                    # compute mean euclidian distance between real and inferred codes
                    mean_euclidian += (code - real_codes).pow(2).sum(1).sqrt().mean().to("cpu").numpy()
                    
                    # compute the hamming distance between real and inferred codes
                    code[code>0] = 1
                    code[code<=0] = -1

                    hamming_dist = F.relu( (code * real_codes) * (-1) ).sum(1)
                    mean_hamming +=  hamming_dist.mean().to("cpu").numpy()

            # save avg euclidiean and hamming distance for displaying at the end of training 
            avg_hamming = mean_hamming / (i_batch+1)
            avg_eucl = mean_euclidian / (i_batch+1)
            self.avg_hamming_dist.append(avg_hamming)
            self.avg_euclidean_dist.append(avg_eucl)

            # now we need to compute the hamming and euclidean distance for the validation set 
            hamming_dist_validation, euclidean_dist_validation, val_loss = self.ecoc_eval_metrics(validation_loader)
            self.validation_loss.append(val_loss)
            self.val_hamming_dist.append(hamming_dist_validation)
            self.val_euclidean_dist.append(euclidean_dist_validation)
            self.train_loss.append(running_loss / (i_batch+1))

            # print epoch metrics 
            print("Epoch: {}, epoch time: {:.4f}, train loss: {:.4f}, val loss: {:.4f}, train ham: {:.4f}, train eucl: {:.4f}, val ham: {:.4f}, val eucl: {:.4f} ".format(i_epoch+1+len_list ,(time.time() - time_0)/60, running_loss / (i_batch+1), val_loss, avg_hamming, avg_eucl, hamming_dist_validation, euclidean_dist_validation)) 

            # save checkpoint of model and metrics just in case smtg goes wrong 
            if (i_epoch+1) % save_iter == 0 : 
                fileName = path +'epoch_'+ str(i_epoch+1+len_list) + '.pth'
                save_load_interface.save_checkpoint(self, fileName, is_indAdvt=True)
        print("Total training Time: {}".format((time.time()-total_time)/60))
   
    def ecoc_eval_metrics(self, dataLoader): 
        '''
        Perform inference of ecoc model on test data and compute metrics such as hamming and euclidean distance between infered and true codes.

                Parameters:
                        dataLoader (DataLoader): data 

                Returns:
                        score, hamming_dist_list, euclidean_dist_list, loss 
        '''
        self.model.eval()

        hamming_dist_list = 0
        euclidean_dist_list = 0
        running_loss = 0
        for i_batch, (images, labels) in enumerate(dataLoader): 
            images, labels = images.to(self.device), labels.to(self.device) 

            # compute outputs of model 
            code = self.model(images)     

            # get real codes from labels and codeword matrix
            real_codes = self.W[labels,:]

            # new optimized hinge loss 
            avg_loss = self.criterion(code, real_codes)          
            running_loss += avg_loss.item()

            with torch.no_grad():
                code = torch.tanh(code)  

                # compute mean euclidian distance between real and inferred codes
                euclidean_dist_list += (code - real_codes).pow(2).sum(1).sqrt().mean().to("cpu").numpy()
                
                # compute the hamming distance between real and inferred codes
                code[code>0] = 1
                code[code<=0] = -1

                hamming_dist = F.relu( (code * real_codes) * (-1) ).sum(1) 
                hamming_dist_list += hamming_dist.mean().to("cpu").numpy()      
        return  hamming_dist_list / (i_batch+1), euclidean_dist_list / (i_batch+1), running_loss / (i_batch+1)
    

    def predict_eval(self, dataLoader):
        '''
        Compute the predictins of the model on some data for an ecoc model when the model is in eval mode 

                Parameters:
                        dataLoader (DataLoader): data 

                Returns:
                        Predictions from the model 
        '''
        self.model.eval()
        X = np.array([])
        y = np.array([])
        with torch.no_grad(): 
            for images, labels in dataLoader: 
                # transfer data on GPU 
                images, labels = images.to(self.device), labels.to(self.device)
                # compute predictions of model 
                outputs = self.model.forward(images)
                _, predictions = torch.max(outputs, 1)
                # compute correct predictions according to labels
                X = np.append(X, predictions.to("cpu").numpy())
                y =  np.append(y, labels.to("cpu").numpy())
        return X, y
    

    def assess_codes(self, dataLoader):
        '''
        Compute the predictins of the model on some data for an ecoc model when the model is in eval mode 

                Parameters:
                        dataLoader (DataLoader): data 

                Returns:
                        Predictions from the model 
        '''
        # separate ECOC members and ECOC decoder 
        ecoc = self.model[0]
        decoder = self.model[1]

        self.model.eval()
        labels_list = torch.empty([0])
        pred_labels_list = torch.empty([0])
        true_codes_list = torch.empty([0])
        real_inferred_codes_list = torch.empty([0])
        inferred_codes_list = torch.empty([0])


        with torch.no_grad(): 
            for images, labels in dataLoader: 
                # transfer data on GPU 
                images, labels = images.to(self.device), labels.to(self.device)

                # compute predictions of model 
                raw_codes = ecoc(images) 
                real_codes = self.W[labels.long(),:]
                logits = decoder(raw_codes) 
                _, pred_labels = torch.max(logits, 1)
                real_inferred_codes = self.W[pred_labels.long()]

                # filter infered codes 
                filtered_codes = raw_codes.clone() 
                filtered_codes[filtered_codes>0] = 1
                filtered_codes[filtered_codes<0] = -1
             
                # compute correct predictions according to labels
                labels_list = torch.cat([labels_list, labels.to("cpu")], dim=0)
                pred_labels_list = torch.cat([pred_labels_list, pred_labels.to("cpu")], dim=0)
                true_codes_list = torch.cat([true_codes_list, real_codes.to("cpu")], dim=0)
                real_inferred_codes_list = torch.cat([real_inferred_codes_list, real_inferred_codes.to("cpu")], dim=0)
                inferred_codes_list = torch.cat([inferred_codes_list, filtered_codes.to("cpu")], dim=0)
        
        return labels_list, pred_labels_list, true_codes_list, inferred_codes_list, real_inferred_codes_list
    

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
    
    def get_hamming_distance(self): 
        '''
                Returns: list of hamming distance at each epoch saved during training  
        '''
        return self.avg_hamming_dist

    def get_euclidean_distance(self): 
        '''
                Returns: list of euclidean distance at each epoch saved during training  
        '''
        return self.avg_euclidean_dist

    def get_val_hamming_distance(self): 
        '''
                Returns: list of validation hamming distance at each epoch saved during training  
        '''
        return self.val_hamming_dist

    def get_val_euclidean_distance(self): 
        '''
                Returns: list of validation euclidean distance at each epoch saved during training  
        '''
        return self.val_euclidean_dist


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

    def set_history(self, train_accuracy, validation_accuracy, train_loss, val_loss, euclidean_dist, hamming_dist, val_hamming_dist, val_euclidean_dist): 
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
        self.validation_loss = val_loss
        self.train_loss = train_loss
        self.avg_euclidean_dist = euclidean_dist
        self.avg_hamming_dist = hamming_dist
        self.val_euclidean_dist = val_euclidean_dist 
        self.val_hamming_dist = val_hamming_dist
