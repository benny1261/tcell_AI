# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:02:07 2022

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
# import torch.nn.functional as F
import torch.optim as optim
import copy
import cv2

class_num = 1

class HybridSN(nn.Module):  
  def __init__(self, in_channels=1, out_channels=class_num):
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
 
  def forward(self, x):
    x = self.conv3d_features(x)
    x = x.view(x.size()[0],x.size()[1]*x.size()[2],x.size()[3],x.size()[4])
    x = self.conv2d_features(x)
    x = x.view(x.size()[0],-1)
    x = self.classifier(x)
    return x

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test

  
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
    
def HDRtoMat(Path , hdr_filename , label_filename ):
    
    #hdr to mat
    imgData= np.fromfile(Path + '/' + hdr_filename , dtype = 'float32')
    # 利用numpy中array的reshape函式將讀取到的資料進行重新排列。
    imgData = imgData.reshape(1088, 2048, 150)
    imgData_2 = imgData[768:868 ,1644:1744 , :]
    imgData_2 = 65535 *  imgData_2 # Now scale by 255
    imgData_2 = imgData_2.astype(np.uint16)
    
    sio.savemat(Path + '/indian_pines_corrected.mat', {'indian_pines_corrected': imgData_2})

    #label to mat
    img = cv2.imread(Path + '/' + label_filename)
    img_2 = img[768:868 ,1644:1744 , 0]
    #cv2.imshow('img' , img_2)
    #cv2.waitKey(0)
    #ret, img_3 = cv2.threshold(img_2,10,1,cv2.THRESH_TRUNC)
    
    sio.savemat(Path + '/Indian_pines_gt.mat', {'indian_pines_gt': img_2})

if __name__ == '__main__':
    
    class_num = 2
    Path = r'C:\Users\smile\Desktop\HybridSN\Data\WBC_HCT116'
    hdr_filename = 'WBC_HCT116_1.raw'
    label_filename = 'WBC_HCT116_1_mask.png'
    HDRtoMat(Path , hdr_filename , label_filename )
    
    X = sio.loadmat(Path + '/indian_pines_corrected.mat')['indian_pines_corrected']
    y = sio.loadmat(Path + '/indian_pines_gt.mat')['indian_pines_gt']
    
    test_ratio = 0.90
    patch_size = 25
    pca_components = 30
    
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)
    
    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)
    
    print('\n... ... create data cubes ... ...')
    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)
    
    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)
    
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest  = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape) 
    print('before transpose: Xtest  shape: ', Xtest.shape) 
    
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest  = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape) 
    print('after transpose: Xtest  shape: ', Xtest.shape) 
    
    trainset = TrainDS()
    testset  = TestDS()
    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=128, shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net = HybridSN().to(device)
    net,current_loss_his,current_Acc_his = train(net)
    
    net.eval()   
    count = 0
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test =  outputs
            count = 1
        else:
            y_pred_test = np.concatenate( (y_pred_test, outputs) )
    
    classification = classification_report(ytest, y_pred_test, digits=4)
    print(classification)
    
    # load the original image
 
    X = sio.loadmat(Path + '/indian_pines_corrected.mat')['indian_pines_corrected']
    y = sio.loadmat(Path + '/indian_pines_gt.mat')['indian_pines_gt']
    
    height = y.shape[0]
    width = y.shape[1]
    
    X = applyPCA(X, numComponents= pca_components)
    X = padWithZeros(X, patch_size//2)
    
    outputs = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            if int(y[i,j]) == 0:
                continue
            else :
                image_patch = X[i:i+patch_size, j:j+patch_size, :]
                image_patch = image_patch.reshape(1,image_patch.shape[0],image_patch.shape[1], image_patch.shape[2], 1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)                                   
                prediction = net(X_test_image)
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i][j] = prediction+1
        if i % 20 == 0:
            print('... ... row ', i, ' handling ... ...')
            
    
    predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(5,5))

    ret, img_3 = cv2.threshold(outputs,2,0,cv2.THRESH_BINARY_INV)
    mix_arr = np.array([outputs,outputs,outputs])
    plt.imsave('./filename.png', outputs)
    
  