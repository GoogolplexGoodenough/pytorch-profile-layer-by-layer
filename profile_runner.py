import torch
import torch.nn as nn
import torchvision.models as models
from profile import profile
import operator
from functools import lru_cache, reduce

def profile_network(network, inputs, warming_up_step = 100, mertrix_step = 1000, save_path = 'res.csv', save_method = 'w'):
    inputs = inputs.cuda()
    @profile
    class net(nn.Module):
        def __init__(self):
            super(net, self).__init__()
            self.model = network

        def forward(self, x):
            return self.model(x)

        def save_res(self, save_method = 'w'):
            self.sum_total_time()
            if save_method not in ['w', 'a']:
                save_method = 'w'
            with open(save_path, save_method) as f:
                if save_method == 'a':
                    f.write('\n')
                f.write(self.__str__())
                f.close()

    n = net().cuda()
    n.start_warming_up()
    for i in range(warming_up_step):
        out = n(inputs)

    n.stop_warming_up()
    n.set_metric_times(mertrix_step)
    out = n(inputs)
    n.save_res(save_method)


if __name__ == "__main__":
    x = torch.randn((1, 3, 224, 224), requires_grad=True).cuda()
    network = models.densenet121(pretrained=True).cuda()
    profile_network(network, x, 1000, 1000)

