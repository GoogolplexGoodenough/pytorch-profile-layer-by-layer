# Pytorch Profiler

If you want to get latency of a network, you can try to use this toolkits to profile it.

It is easy to profile a network and show the result layer by layer. The result is similiar to the result got from torch.autograd.profile, as this is based on it. The result will be saved as a csv file.

This is a demo to show how to use it:

```python
import torch
import torchvision.models as models
from profile_runner import profile_network

x = torch.randn((1, 3, 224, 224), requires_grad=True).cuda() # sample inputs
network = models.alexnet(pretrained = True).cuda()  # alexnet for example, and it can be other network built from nn.Module
profile_network(network = network, inputs = x, warming_up_step = 1000, mertrix_step = 1000, save_path = 'res.csv')
```

And you can use it on your own networks. For example:
```python
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
```

The result is following table. (AlexNet)
```
| Total CPU Time | Total CUDA Time | Total CPU time% | Total CUDA Time % | Average CPU Time | Average CUDA Time | Parameters | Input Size | Input Shape        | Input Data Type | Numbers of Calls | Architecture                                                                           |
|----------------|-----------------|-----------------|-------------------|------------------|-------------------|------------|------------|--------------------|-----------------|------------------|----------------------------------------------------------------------------------------|
| 169\.59ms      | 218\.45ms       | 7\.41%          | 5\.52%            | 169\.59us        | 218\.45us         | 23\.30k    | 150\.53k   | \[1, 3, 224, 224\] | torch\.float32  | 1000             | \(0\): Conv2d\(3, 64, kernel\_size=\(11, 11\), stride=\(4, 4\), padding=\(2, 2\)\)     |
| 36\.35ms       | 41\.26ms        | 1\.59%          | 1\.04%            | 36\.35us         | 41\.26us          | \-         | 193\.60k   | \[1, 64, 55, 55\]  | torch\.float32  | 1000             | \(1\): ReLU\(inplace=True\)                                                            |
| 117\.48ms      | 134\.13ms       | 5\.14%          | 3\.39%            | 117\.48us        | 134\.13us         | \-         | 193\.60k   | \[1, 64, 55, 55\]  | torch\.float32  | 1000             | \(2\): MaxPool2d\(kernel\_size=3, stride=2, padding=0, dilation=1, ceil\_mode=False\)  |
| 268\.16ms      | 343\.54ms       | 11\.72%         | 8\.68%            | 268\.16us        | 343\.54us         | 307\.39k   | 46\.66k    | \[1, 64, 27, 27\]  | torch\.float32  | 1000             | \(3\): Conv2d\(64, 192, kernel\_size=\(5, 5\), stride=\(1, 1\), padding=\(2, 2\)\)     |
| 37\.77ms       | 43\.00ms        | 1\.65%          | 1\.09%            | 37\.77us         | 43\.00us          | \-         | 139\.97k   | \[1, 192, 27, 27\] | torch\.float32  | 1000             | \(4\): ReLU\(inplace=True\)                                                            |
| 111\.26ms      | 124\.99ms       | 4\.86%          | 3\.16%            | 111\.26us        | 124\.99us         | \-         | 139\.97k   | \[1, 192, 27, 27\] | torch\.float32  | 1000             | \(5\): MaxPool2d\(kernel\_size=3, stride=2, padding=0, dilation=1, ceil\_mode=False\)  |
| 179\.42ms      | 222\.62ms       | 7\.84%          | 5\.62%            | 179\.42us        | 222\.62us         | 663\.94k   | 32\.45k    | \[1, 192, 13, 13\] | torch\.float32  | 1000             | \(6\): Conv2d\(192, 384, kernel\_size=\(3, 3\), stride=\(1, 1\), padding=\(1, 1\)\)    |
| 37\.81ms       | 42\.85ms        | 1\.65%          | 1\.08%            | 37\.81us         | 42\.85us          | \-         | 64\.90k    | \[1, 384, 13, 13\] | torch\.float32  | 1000             | \(7\): ReLU\(inplace=True\)                                                            |
| 199\.94ms      | 266\.91ms       | 8\.74%          | 6\.74%            | 199\.94us        | 266\.91us         | 884\.99k   | 64\.90k    | \[1, 384, 13, 13\] | torch\.float32  | 1000             | \(8\): Conv2d\(384, 256, kernel\_size=\(3, 3\), stride=\(1, 1\), padding=\(1, 1\)\)    |
| 37\.14ms       | 42\.21ms        | 1\.62%          | 1\.07%            | 37\.14us         | 42\.21us          | \-         | 43\.26k    | \[1, 256, 13, 13\] | torch\.float32  | 1000             | \(9\): ReLU\(inplace=True\)                                                            |
| 188\.75ms      | 239\.40ms       | 8\.25%          | 6\.05%            | 188\.75us        | 239\.40us         | 590\.08k   | 43\.26k    | \[1, 256, 13, 13\] | torch\.float32  | 1000             | \(10\): Conv2d\(256, 256, kernel\_size=\(3, 3\), stride=\(1, 1\), padding=\(1, 1\)\)   |
| 36\.70ms       | 41\.58ms        | 1\.61%          | 1\.05%            | 36\.70us         | 41\.58us          | \-         | 43\.26k    | \[1, 256, 13, 13\] | torch\.float32  | 1000             | \(11\): ReLU\(inplace=True\)                                                           |
| 105\.73ms      | 116\.84ms       | 4\.62%          | 2\.95%            | 105\.73us        | 116\.84us         | \-         | 43\.26k    | \[1, 256, 13, 13\] | torch\.float32  | 1000             | \(12\): MaxPool2d\(kernel\_size=3, stride=2, padding=0, dilation=1, ceil\_mode=False\) |
| 109\.53ms      | 123\.03ms       | 4\.79%          | 3\.11%            | 109\.53us        | 123\.03us         | \-         | 9\.22k     | \[1, 256, 6, 6\]   | torch\.float32  | 1000             | \(avgpool\): AdaptiveAvgPool2d\(output\_size=\(6, 6\)\)                                |
| 81\.93ms       | 92\.68ms        | 3\.58%          | 2\.34%            | 81\.93us         | 92\.68us          | \-         | 9\.22k     | \[1, 9216\]        | torch\.float32  | 1000             | \(0\): Dropout\(p=0\.5, inplace=False\)                                                |
| 142\.29ms      | 1\.10s          | 6\.22%          | 27\.83%           | 142\.29us        | 1\.10ms           | 37\.75m    | 9\.22k     | \[1, 9216\]        | torch\.float32  | 1000             | \(1\): Linear\(in\_features=9216, out\_features=4096, bias=True\)                      |
| 38\.11ms       | 52\.08ms        | 1\.67%          | 1\.32%            | 38\.11us         | 52\.08us          | \-         | 4\.10k     | \[1, 4096\]        | torch\.float32  | 1000             | \(2\): ReLU\(inplace=True\)                                                            |
| 82\.98ms       | 93\.78ms        | 3\.63%          | 2\.37%            | 82\.98us         | 93\.78us          | \-         | 4\.10k     | \[1, 4096\]        | torch\.float32  | 1000             | \(3\): Dropout\(p=0\.5, inplace=False\)                                                |
| 186\.15ms      | 436\.75ms       | 8\.14%          | 11\.03%           | 186\.15us        | 436\.75us         | 16\.78m    | 4\.10k     | \[1, 4096\]        | torch\.float32  | 1000             | \(4\): Linear\(in\_features=4096, out\_features=4096, bias=True\)                      |
| 36\.61ms       | 41\.52ms        | 1\.60%          | 1\.05%            | 36\.61us         | 41\.52us          | \-         | 4\.10k     | \[1, 4096\]        | torch\.float32  | 1000             | \(5\): ReLU\(inplace=True\)                                                            |
| 83\.85ms       | 139\.91ms       | 3\.67%          | 3\.53%            | 83\.85us         | 139\.91us         | 4\.10m     | 4\.10k     | \[1, 4096\]        | torch\.float32  | 1000             | \(6\): Linear\(in\_features=4096, out\_features=1000, bias=True\)                      |

```
