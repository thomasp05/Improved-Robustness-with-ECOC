import numpy as np 
import torch

def predict_eval(model, dataLoader, device):
        '''
        Compute the predictins of the model on some data for an ecoc model when the model is in eval mode 

                Parameters:
                        dataLoader (DataLoader): data 

                Returns:
                        Predictions from the model 
        '''
        model.eval()
        X = np.array([])
        y = np.array([])
        with torch.no_grad(): 
            for images, labels in dataLoader: 
                # transfer data on GPU 
                images, labels = images.to(device), labels.to(device)
                # compute predictions of model 
                outputs = model.forward(images)
                _, predictions = torch.max(outputs, 1)
                # compute correct predictions according to labels
                X = np.append(X, predictions.to("cpu").numpy())
                y =  np.append(y, labels.to("cpu").numpy())
        return X, y


# compute the accuracy in percentage of predictions X with true labels y 
def success_rate(pred_clean, true_labels, pred_perturbed):    
    
    good_predictions = np.where(pred_clean == true_labels)[0]
    pred_perturbed_good = pred_perturbed[good_predictions] 
    true_labels_good = true_labels[good_predictions]
    asr = (pred_perturbed_good == true_labels_good).sum() / good_predictions.shape[0] * 100
    return asr


# compute the accuracy in percentage of predictions X with true labels y 
def score(X, y):    
    length = y.shape[0]
    score = np.sum(X == y) / length * 100 
    return score
    
def score_stats(X, y):    
    length = y.shape[0]
    good = np.where(X==y) 
    good = y[good]

    totals = [] 
    count = []
    for i in range(10): 
        totals.append(np.sum(y == i)) 
        count.append(np.sum(good == i))
    perc = [i / j *100 for i, j in zip(count, totals)] 
    print(perc) 
    score = np.sum(X == y) / length * 100 
    return score

