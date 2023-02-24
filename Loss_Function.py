
import torch
import torch.nn.functional as F 

# optimized implementation of hinge loss
class hinge_vec(torch.nn.Module): 
    def __init__(self): 
        super(hinge_vec, self).__init__()

    def forward(self, preds, labels): 
        loss = (1 - preds * labels) 
        loss = F.relu(loss)
        loss = loss.sum(1) 
        loss = torch.mean(loss)
        return loss

# CW loss function from  "Towards Evaluating the Robustness of Neural Networks"
def cw_loss(logits, labels):
        label_mask = F.one_hot(labels, num_classes=logits.shape[1])
        correct_logit = torch.sum(label_mask * logits, axis=1)
        wrong_logit = torch.max((1-label_mask) * logits - 1e4*label_mask, axis=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + 50)          
        return loss.mean()