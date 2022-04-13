import torch
import torch.nn as nn
import torch.nn.functional as F


def getBinaryTensor_2(input):
    one = torch.ones_like(input)
    zero = torch.zeros_like(input)
    boundary = (torch.max(input) + torch.min(input))/2
    return torch.where(input > boundary, one, zero)


class breast_client(nn.Module):
    def __init__(self, learning_rate=None, split_point=6, type='client'):
        super(breast_client, self).__init__()
        self.w1 = torch.normal(mean=0., std=0.1, size=(9, 64), requires_grad=True)
        self.b1 = torch.zeros(size=(64,), requires_grad=True)
        self.w2 = torch.normal(mean=0., std=0.1, size=(64, 128), requires_grad=True)
        self.b2 = torch.zeros(size=(128,), requires_grad=True)
        self.w3 = torch.normal(mean=0., std=0.1, size=(128, 258), requires_grad=True)
        self.b3 = torch.zeros(size=(258,), requires_grad=True)
        self.w4 = torch.normal(mean=0., std=0.1, size=(258, 512), requires_grad=True)
        self.b4 = torch.zeros(size=(512,), requires_grad=True)
        self.w5 = torch.normal(mean=0., std=0.1, size=(512, 128), requires_grad=True)
        self.b5 = torch.zeros(size=(128,), requires_grad=True)
        self.w6 = torch.normal(mean=0., std=0.1, size=(128, 1), requires_grad=True)
        self.b6 = torch.zeros(size=(1,))
        self.weights = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3,
                        self.w4, self.b4, self.w5, self.b5, self.w6, self.b6]
        self.layer_num = 6
        self.split_point = split_point
        self.type = type

    def forward(self, x):

        if self.type == 'client':
            for i in range(0, self.split_point*2, 2):
                x = torch.mm(x, self.weights[i]) + self.weights[i+1]
                if i != self.layer_num*2-2:
                    x = F.sigmoid(x)
                else:
                    x = F.sigmoid(x)
        else:
            for i in range(self.split_point*2, self.layer_num*2, 2):
                x = torch.mm(x, self.weights[i]) + self.weights[i+1]
                if i != self.layer_num*2-2:
                    x = F.sigmoid(x)
                else:
                    x = F.sigmoid(x)

        return x
