import torch.nn as nn
import torch.nn.functional as F

from variational_dropout import VariationalDropout

DIM_IMG = 32

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        conv_ksize = [5, 5]
        conv_outC = [6, 16]
        conv_wid1 = DIM_IMG - conv_ksize[0] + 1
        conv_wid2 = conv_wid1 - conv_ksize[1] + 1
        self.conv1 = nn.Conv2d(3, conv_outC[0], conv_ksize[0])
        #self.conv1_vardrop = VariationalDropout( conv_outC[0]*conv_wid1*conv_wid1, conv_outC[0]*conv_wid1*conv_wid1 )
        self.conv2 = nn.Conv2d(6, conv_outC[1], conv_ksize[1])
        #self.conv2_vardrop = VariationalDropout( conv_outC[1]*conv_wid2*conv_wid2, conv_outC[1]*conv_wid2*conv_wid2 )
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc1_vardrop = VariationalDropout(120, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, train=True):
        C1 = F.relu(self.conv1(x))
        S2 = F.max_pool2d(C1, 2)
        C3 = F.relu(self.conv2(S2))
        S4 = F.max_pool2d(C3,2)
        S4_Flat = S4.view(S4.size(0),-1)
        F1 = F.relu(self.fc1(S4_Flat))
        if train:
            VD, self.kld = self.fc1_vardrop(F1, train)
        else:
            VD = self.fc1_vardrop(F1, train)
        F2 = F.relu(self.fc2(VD))
        F3 = self.fc3(F2)
        return x, C1, S2, C3, S4, F1, F2, F3

    def loss(self, **kwargs):
        if kwargs['train']:
            kld = self.kld
            x, C1, S2, C3, S4, F1, F2, F3 = self(kwargs['input'], kwargs['train'])
            return F.cross_entropy(F3, kwargs['target'], size_average=kwargs['average']),kld
        x, C1, S2, C3, S4, F1, F2, F3 = self(kwargs['input'], kwargs['train'])
        return F.cross_entropy(F3, kwargs['target'], size_average=kwargs['average'])