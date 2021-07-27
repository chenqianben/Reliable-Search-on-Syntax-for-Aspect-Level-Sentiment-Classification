import torch
import torch.nn as nn

class CrossEntropy(nn.Module):
    ''' Cross-entropy loss with syntactic regularization term and label-smoothing '''
    def __init__(self, beta, eps, gamma):
        super(CrossEntropy, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.confidence = 1.0 - eps
        self.smoothing = eps
        self.classes = 3
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, outputs, labels):
        predicts, weight_norm = outputs
        predicts = self.logsoftmax(predicts)
        cur_beta = self.beta * torch.ones_like(labels) + \
                   (self.beta*(self.gamma-1)) * torch.where(labels == 2, torch.ones_like(labels), torch.zeros_like(labels))
        with torch.no_grad():
            true_dist = torch.zeros_like(predicts)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, labels.data.unsqueeze(1), self.confidence)
        return -torch.mean(torch.sum(true_dist * predicts, dim=-1)) + torch.mean(cur_beta * torch.pow(weight_norm, 2))
    
