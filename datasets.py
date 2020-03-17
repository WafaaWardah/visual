'''
    Written by Wafaa Wardah  | USP    | June 2019
    Data set class for PyTorch

    Creating an object of this class requires the prepared prepared dataset in csv format
'''

import torch
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ProtPep_dataset(Dataset):
    
    def __init__(self, ws, mode):
        
        self.ws = ws
        self.mode = mode
        
        input_file = self.mode + '_' + str(self.ws) + '_set.csv'
        label_file = str(self.mode) + '_labels.txt'
        
        self.image_list = []
        self.label_list = []
        
        with open(input_file, 'r') as input_f:
            
            buffer = csv.reader(input_f)
            img = []
            for i, row in enumerate(buffer):
                row = [float(n) for n in row]
                img.append(row)
                
                if (i+1)%self.ws == 0:
                    self.image_list.append(np.array(img))
                    img.clear()
        
        with open(label_file, 'r') as label_f:
            self.label_list = list(label_f.read())
            
            
    def shuffle_lists(self, l1, l2):
        random.seed(4)
        mapIndexPosition = list(zip(l1, l2))
        random.shuffle(mapIndexPosition)
        l1, l2 = zip(*mapIndexPosition)
        return list(l1), list(l2)
    
        
    def __getitem__(self, index):
        #plt.imshow(self.image_list[index], 'gray')
        #plt.savefig('image.png')
        #plt.show()
        return torch.tensor(self.image_list[index]), torch.tensor(int(self.label_list[index]))
    
    
    def __len__(self):
        return len(self.label_list)

