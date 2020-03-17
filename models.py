'''
    Written by Wafaa Wardah  | USP    | June 2019
    Model class for PyTorch 
'''

import torch.nn as nn

class dynamic_model(nn.Module):
    def __init__(self, H_in, W_in, num_kernels):
        super(dynamic_model, self).__init__()
        
        C_in_1, C_out_1     = 1, num_kernels
        kernel_size_1       = 3
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(int(C_in_1), int(C_out_1), kernel_size=kernel_size_1, stride=1, padding=0),
            nn.ReLU())
        
        H_out_1, W_out_1    = self.output_shape((H_in, W_in), kernel_size=kernel_size_1) # W_in = 38
        C_in_2, C_out_2     = C_out_1, num_kernels
        kernel_size_2       = 2
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(int(C_in_2), int(C_out_2), kernel_size=kernel_size_2, stride=1, padding=0),
            nn.ReLU())        
        
        H_out_2, W_out_2    = self.output_shape((H_out_1, W_out_1), kernel_size=kernel_size_2)
        
        # for Maxpooling layer
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        H_out_Mx, W_out_Mx    = self.output_shape((H_out_2, W_out_2), kernel_size=2, stride=2)
        self.fc1 = nn.Sequential(nn.Linear(int(C_out_2) * int(H_out_Mx) * int(W_out_Mx), 2))
            
    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        from math import floor
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
        w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
        return h, w
