import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, dropout=0.01):
        super(Net, self).__init__()

        ## Convolution Block 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),   # Input: 32x32x32 | Output: 32x32x32 | RF: 5x5
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        ## Transition Block1

        self.trans1=  nn.Sequential(
            #Stride 2 conv
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, bias=False),  # Input: 32x32x32 | Output: 16x16x32 | RF: 7x7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

        )

        ## Convolution Block 2

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),   # Input: 16x16x32 | Output: 16x16x64 | RF: 11x11
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),  

        ) 

        ## Transiton Block 2
        self.trans2 =  nn.Sequential(
            #Adding Depthwise Convolution with stride2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=2, groups=64, bias=False),
            #pointwise
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,1)), # Input: 16x16x64 | Output: 8x8x64 | RF: 15x15
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        ## Convolution Block 3

        self.conv3 = nn.Sequential(

            # Adding depthwise convolution
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, groups=64, bias=False),
            #pointwise
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1)), # Input: 8x8x64 | Output: 8x8x128 | RF: 23x23
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout),  
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False), # Input: 8x8x128 | Output: 8x8x32 | RF: 23x23
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout), 

        ) 

        self.trans3 =  nn.Sequential(
            ## Adding two layers of dilation
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False), # Input: 8x8x128 | Output: 6x6x32 | RF: 39x39
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False), # Input: 6x6x32 | Output: 4x4x32 | RF: 55x55
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), # Input: 4x4x32 | Output: 4x4x32 | RF: 63x63
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(3, 3), padding=1, bias=False), # Input: 4x4x32 | Output: 4x4x10 | RF: 71x71

        ) 

        self.gap = nn.AvgPool2d(4)

    def forward(self, x):
        x =  self.trans1(self.conv1(x))
        x =  self.trans2(self.conv2(x))
        x =  self.trans3(self.conv3(x))
        x =  self.conv4(x)
        x =  self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)