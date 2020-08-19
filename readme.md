# Pytorch Profiler

If you want to get latency of a network, you can try to use this toolkits to profile it.

It is easy to profile a network and show the result layer by layer. The result is similiar to the result got from torch.autograd.profile, as this is based on it. The result will be saved as a csv file.

This is a demo to show how to use it:

'''python
import torch
import torchvision.models as models
from profile_runner import profile_network

x = torch.randn((1, 3, 224, 224), requires_grad=True).cuda() # sample inputs
network = models.alexnet(pretrained = True).cuda()  # alexnet for example, and it can be other network built from nn.Module
profile_network(network = network, inputs = x, warming_up_step = 1000, mertrix_step = 1000, save_path = 'res.csv')
'''

And you can use it on your own networks. For example:
'''python
import torch
import torch.nn as nn
from profile_runner import profile_network

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv = nn.Conv2d(3, 64, 11, 4, 2)
        list = [nn.Conv2d(64,64,3,1,1) for _ in range(5)]
        list.append(nn.ReLU(True))
        self.body = nn.Sequential(*list)

    def forward(self, x):
        return self.body(self.conv(x))

x = torch.randn((1, 3, 224, 224), requires_grad=True).cuda() # sample inputs
network = Net1().cuda()  # alexnet for example, and it can be other network built from nn.Module
profile_network(network = network, inputs = x, warming_up_step = 1000, mertrix_step = 1000, save_path = 'res.csv')
'''