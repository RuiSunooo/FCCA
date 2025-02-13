import pickle
import torch
from fast_dfew import  fasternet_s
from torch.autograd import Variable
import torch.nn as nn

class Creatmodel(nn.Module):

    def __init__(self, args, pretrained=True):
        super(Creatmodel, self).__init__()
        self.fasternet = fasternet_s(pretrained=True)

        self.features = nn.Sequential(*list(self.fasternet.children())[:-1])
        self.unpooling = torch.nn.Upsample(scale_factor=7, mode='nearest')

        self.features2 = nn.Sequential(*list(self.fasternet.children())[-1:-1])

        self.fc = nn.Linear(20480, 7)

    def forward(self, x):
        # print('x0:',x.shape)
        x = x.contiguous().view(-1, 3, 112, 112)
        # print('x1:', x.shape)
        x = self.features(x)
        # print('x.shape', x.shape)
        b,c,h,w = x.shape
        x= x.reshape(b//16,16,c,h,w)

        #### 1, 2048, 7, 7
        feature = self.features2(x)
        # print('feature.shape',feature.shape)
        #### 1, 2048, 1, 1



        feature = feature.view(feature.size(0), -1)
        # print('feature.shape1', feature.shape)



        output = self.fc(feature)

        params = list(self.parameters())
        fc_weights = params[-2].data
        fc_weights = fc_weights.view(1, 7, 20480, 1, 1)
        fc_weights = Variable(fc_weights, requires_grad=False)

        # attention
        feat = self.unpooling(x)
        feat = feat.unsqueeze(1)  # N * 1 * C * H * W
        feat =feat.view(40,7,20480,7,7)
        # print('feat.shape',feat.shape)
        # print('fc_weights.shape',fc_weights.shape)
        hm = feat * fc_weights

        hm = hm.sum(2)  # N * self.num_labels * H * W

        return feature,output, hm,