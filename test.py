import numpy as np
import torch
import torch.nn as nn
import cv2

class HybridSN(nn.Module):  
    def __init__(self, in_channels=1, out_channels= 1):
        super(HybridSN, self).__init__()
        self.conv3d_features = nn.Sequential(
        nn.Conv3d(in_channels,out_channels=8,kernel_size=(7,3,3)),
        nn.ReLU(),
        nn.Conv3d(in_channels=8,out_channels=16,kernel_size=(5,3,3)),
        nn.ReLU(),
        nn.Conv3d(in_channels=16,out_channels=32,kernel_size=(3,3,3)),
        nn.ReLU()
        )

        self.conv2d_features = nn.Sequential(
        nn.Conv2d(in_channels=32 * 18, out_channels=64, kernel_size=(3,3)),
        nn.ReLU()
        )

        self.classifier = nn.Sequential(
        nn.Linear(64 * 17 * 17, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(128, 16)
        )
 

def tst(t):
    if t >= 3 and t % 2 == 1:
        print('yes')
    else:
        print('no')

def padWithZeros(x, margin):
    new_x = np.zeros((x.shape[0] + 2 * margin, x.shape[1] + 2* margin)) # creating a zero matrix with size expanded from x
    new_x[ margin: x.shape[0] + margin, margin: x.shape[1] + margin] = x
    return new_x

# t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# t = np.array(t)
# t.reshape((3, 3, 1))
# print(t)
# print(t.shape)

# mask_file = 'data/B1_merged_color_mask.png'
# img = cv2.imread(mask_file)
# print(img.shape)
# print(img)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = HybridSN().to(device)
net.eval()
net = net.cuda()