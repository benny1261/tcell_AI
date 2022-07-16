# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:02:07 2022

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import cv2
import os


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
 
    # def forward(self, x):
    #     x = self.conv3d_features(x)
    #     x = x.view(x.size()[0],x.size()[1]*x.size()[2],x.size()[3],x.size()[4])
    #     x = self.conv2d_features(x)
    #     x = x.view(x.size()[0],-1)
    #     x = self.classifier(x)
    #     return x

def applyPCA(x, numComponents = 30):
    x_channal = np.reshape(x, (-1, x.shape[2]))
    pca = PCA(n_components= numComponents, whiten=True)
    x_pca = pca.fit_transform(x_channal)
    return np.reshape(x_pca, (x.shape[0], x.shape[1], numComponents))

def padWithZeros(x, margin):
    new_x = np.zeros((x.shape[0] + 2 * margin, x.shape[1] + 2* margin, x.shape[2])) # creating a zero matrix with size expanded from x
    new_x[ margin: x.shape[0] + margin, margin: x.shape[1] + margin, :] = x
    return new_x

def createImageCubes(x, y, windowSize: int= 5, removeZeroLabels = False):
    '''window size should be at least 3 and be an odd number'''
    if windowSize >= 3 and windowSize % 2 == 1:
        # 给 x 做 padding
        margin = int((windowSize - 1) / 2)
        zeroPaddedx = padWithZeros(x, margin)
        # split patches
        patchesData = np.zeros((x.shape[0] * x.shape[1], windowSize, windowSize, x.shape[2]))   # add 2 dimensions, delete space dimension
        patchesLabels = np.zeros((x.shape[0] * x.shape[1], 3))     # second dimension stores RGB channal

        for r in range(x.shape[0]):
            for c in range(x.shape[1]):
                patchIndex = r*x.shape[1] + c
                patchesData[patchIndex, :, :] = zeroPaddedx[r: r + windowSize, c: c + windowSize]
                patchesLabels[patchIndex, :] = y[r, c]
        if removeZeroLabels:
            patchesData = patchesData[patchesLabels>0,:,:,:]
            patchesLabels = patchesLabels[patchesLabels>0]
            patchesLabels -= 1  # ????????????????????????????????????????????????????????
        return patchesData, patchesLabels
    
    else:
        print('window size should be at least 3 and be an odd number')



""" Training dataset"""
class TrainDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        return self.len

   

def train(net):

    current_loss_his = []
    current_Acc_his = []

    best_net_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    total_loss = 0
    for epoch in range(2):
        net.train() 
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        net.eval()   
        current_acc = test_acc(net)
        current_Acc_his.append(current_acc)
        
        if current_acc > best_acc:
            best_acc = current_acc
            best_net_wts = copy.deepcopy(net.state_dict())
            torch.save(net.state_dict(), './Snapshot/Best_model.pth')

        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]  [current acc: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item(), current_acc))
        current_loss_his.append(loss.item())

    print('Finished Training')
    print("Best Acc:%.4f" %(best_acc))

    # load best model weights
    net.load_state_dict(best_net_wts)

    return net,current_loss_his,current_Acc_his

def test_acc(net):
    count = 0
    # 模型测试
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test =  outputs
            count = 1
        else:
            y_pred_test = np.concatenate( (y_pred_test, outputs) )

    # 生成分类报告
    classification = classification_report(ytest, y_pred_test, digits=4)
    index_acc = classification.find('weighted avg')
    accuracy = classification[index_acc+17:index_acc+23]
    return float(accuracy)

if __name__ == '__main__':

# io -----------------------------------------------------------------------------------------------------------------------------
    os.chdir('data')
    raw_file = 'B1_processed.raw'
    mask_file = 'B1_merged_color_mask.png'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = os.getcwd() # +model file

    # saving as .mat dtype
    # print('\ncreating database......')
    # img = cv2.imread(mask_file)
    # sio.savemat('data_y.mat', {'mask': img})

    # vraw = np.fromfile(raw_file , dtype = 'float32')
    # raw = vraw.reshape(img.shape[0], img.shape[1], 150) #(h, w, c)
    # sio.savemat('data_x.mat', {'raw': raw})

# parameters ---------------------------------------------------------------------------------------------------------------------
    class_num = 4
    TEST_RATIO = 0.10
    patch_size = 5 # n*n cube
    pca_components = 30
    split_seed = 1

# loading data and model ---------------------------------------------------------------------------------------------------------
    x = sio.loadmat('data_x.mat')['raw']
    y = sio.loadmat('data_y.mat')['mask']

    # height = y.shape[0]
    # width = y.shape[1]

    net = HybridSN().to(device)
    try:
        net.load_state_dict(torch.load(MODEL_PATH))

    except:
        print("no saved model")
    net.eval()
    net = net.cuda()

# calculation --------------------------------------------------------------------------------------------------------------------
    print('\nPCA transformation......')
    x_pca = applyPCA(x, numComponents= pca_components)
    
    print('\ncreating data cubes......')
    x_cube, y = createImageCubes(x_pca, y, windowSize= patch_size)
    print('Data cube raw shape: ', x_cube.shape)
    print('Data cube mask shape: ', y.shape)

    print('\ncreate train & test data......')
    x_train, x_test, y_train, y_test = train_test_split(x_cube, y, test_size= TEST_RATIO, random_state= split_seed, stratify= y)
    print('train samples: ',y_train.shape[0])
    print('test samples: ',y_test.shape[0])
    
    # x_train = x_train.reshape(-1, patch_size, patch_size, pca_components, 1)
    # x_test  = x_test.reshape(-1, patch_size, patch_size, pca_components, 1)
    
    # Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    # Xtest  = Xtest.transpose(0, 4, 3, 1, 2)
    # print('after transpose: Xtrain shape: ', Xtrain.shape) 
    # print('after transpose: Xtest  shape: ', Xtest.shape) 
    
    # trainset = TrainDS()
    # testset  = TestDS()
    # train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True)
    # test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=128, shuffle=False)
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # net = HybridSN().to(device)
    # net,current_loss_his,current_Acc_his = train(net)
    
    # net.eval()   
    # count = 0
    # for inputs, _ in test_loader:
    #     inputs = inputs.to(device)
    #     outputs = net(inputs)
    #     outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    #     if count == 0:
    #         y_pred_test =  outputs
    #         count = 1
    #     else:
    #         y_pred_test = np.concatenate( (y_pred_test, outputs) )
    
    # classification = classification_report(ytest, y_pred_test, digits=4)
    # print(classification)
    
    # # load the original image
 
    # X = sio.loadmat('/indian_pines_corrected.mat')['indian_pines_corrected']
    # y = sio.loadmat('/indian_pines_gt.mat')['indian_pines_gt']
    

    
    # X = applyPCA(X, numComponents= pca_components)
    # X = padWithZeros(X, patch_size//2)
    
    # outputs = np.zeros((height,width))
    # for i in range(height):
    #     for j in range(width):
    #         if int(y[i,j]) == 0:
    #             continue
    #         else :
    #             image_patch = X[i:i+patch_size, j:j+patch_size, :]
    #             image_patch = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1)
    #             X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)                                   
    #             prediction = net(X_test_image)
    #             prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
    #             outputs[i][j] = prediction+1
    #     if i % 20 == 0:
    #         print('... ... row ', i, ' handling ... ...')
            
    
    # predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(5,5))

    # ret, img_3 = cv2.threshold(outputs,2,0,cv2.THRESH_BINARY_INV)
    # mix_arr = np.array([outputs,outputs,outputs])
    # plt.imsave('./filename.png', outputs)
    
  