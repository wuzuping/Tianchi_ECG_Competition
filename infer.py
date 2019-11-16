#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 20:02:29 2019

@author: wuzuping
"""
import pandas as pd
import numpy as np
import os
from scipy.signal import resample
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from multiprocessing import Pool
from biosppy.signals import ecg
from pyentrp import entropy as ent
import pywt
from senet import se_resnet50,se_resnet101
from resnet import resnet34
from densenet import densenet121

def WTfilt_1d(sig):
    """
    # 使用小波变换对单导联ECG滤波
    # 参考：Martis R J, Acharya U R, Min L C. ECG beat classification using PCA, LDA, ICA and discrete
    wavelet transform[J].Biomedical Signal Processing and Control, 2013, 8(5): 437-448.
    :param sig: 1-D numpy Array，单导联ECG
    :return: 1-D numpy Array，滤波后信号
    """
    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt

class ManFeat_HRV(object):
    """
        针对一条记录的HRV特征提取， 以II导联为基准
    """
    FEAT_DIMENSION = 9

    def __init__(self, sig, fs=250.0):
        assert len(sig.shape) == 1, 'The signal must be 1-dimension.'
        assert sig.shape[0] >= fs * 6, 'The signal must >= 6 seconds.'
        self.sig = WTfilt_1d(sig)
        self.fs = fs
        self.rpeaks, = ecg.hamilton_segmenter(signal=self.sig, sampling_rate=self.fs)
        self.rpeaks, = ecg.correct_rpeaks(signal=self.sig, rpeaks=self.rpeaks,
                                         sampling_rate=self.fs)
        self.RR_intervals = np.diff(self.rpeaks)
        self.dRR = np.diff(self.RR_intervals)
    
    def __get_sdnn(self):  # 计算RR间期标准差
        return np.array([np.std(self.RR_intervals)])

    def __get_maxRR(self):  # 计算最大RR间期
        return np.array([np.max(self.RR_intervals)])

    def __get_minRR(self):  # 计算最小RR间期
        return np.array([np.min(self.RR_intervals)])

    def __get_meanRR(self):  # 计算平均RR间期
        return np.array([np.mean(self.RR_intervals)])

    def __get_Rdensity(self):  # 计算R波密度
        return np.array([(self.RR_intervals.shape[0] + 1) 
                         / self.sig.shape[0] * self.fs])

    def __get_pNN50(self):  # 计算pNN50
        return np.array([self.dRR[self.dRR >= self.fs*0.05].shape[0] 
                         / self.RR_intervals.shape[0]])

    def __get_RMSSD(self):  # 计算RMSSD
        return np.array([np.sqrt(np.mean(self.dRR*self.dRR))])
    
    def __get_SampEn(self):  # 计算RR间期采样熵
        sampEn = ent.sample_entropy(self.RR_intervals, 
                                  2, 0.2 * np.std(self.RR_intervals))
        for i in range(len(sampEn)):
            if np.isnan(sampEn[i]):
                sampEn[i] = -2
            if np.isinf(sampEn[i]):
                sampEn[i] = -1
        return sampEn

    def extract_features(self):  # 提取HRV所有特征
        if len(self.RR_intervals) == 0 or len(self.dRR) == 0:
            features =  np.zeros((ManFeat_HRV.FEAT_DIMENSION,))
        else:
            features = np.concatenate((self.__get_sdnn(),
                    self.__get_maxRR(),
                    self.__get_minRR(),
                    self.__get_meanRR(),
                    self.__get_Rdensity(),
                    self.__get_pNN50(),
                    self.__get_RMSSD(),
                    self.__get_SampEn(),
                    ))
        assert features.shape[0] == ManFeat_HRV.FEAT_DIMENSION
        return features
    
def get_RR(data):
    result = []
    for i in range(12):
        arr = data[:,i]
        s = ManFeat_HRV(arr, fs=250)
        fea = s.extract_features()
        result.append(fea)
    result = np.concatenate(result)
    return result

def RR_Feat(Id):
    data = pd.read_csv('../tcdata/hf_round2_testB/'+Id, sep=' ')
    data['III'] = data['II']-data['I']
    data['aVR']=-(data['II']+data['I'])/2
    data['aVL']=(data['I']-data['II'])/2
    data['aVF']=(data['II']-data['I'])/2
    data = data.astype('float').values
    data = resample(data, 2500)
    feas = get_RR(data)
    return feas

arrythmia = pd.read_csv('hf_round2_arrythmia.txt',header=None,sep="\t")#读入心电异常事件列表
label_map = dict(zip(arrythmia[0].unique(), range(arrythmia.shape[0])))

class ToTensor(object):
    """
    convert ndarrays in sample to Tensors.
    return:
        feat(torch.FloatTensor)
        label(torch.LongTensor of size batch_size x 1)
    """
    def __call__(self, data):
        data = torch.from_numpy(data).type(torch.FloatTensor)
        return data
    
subA = pd.read_csv('../tcdata/hf_round2_subB.txt', sep='\t', names=['id', 'age', 'gender'])
subA['age'] = subA['age'].fillna(0.0)
subA['gender'] = subA['gender'].fillna('MISS')
with Pool(8) as p:
    Feat_RR = p.map(RR_Feat, list(subA['id']))
Feat_RR = np.stack(Feat_RR)

class TestData(Dataset):
    def __init__(self, fname,mode,feat, transform=None):
        self.fname = fname
        self.transform = transform
        self.mode=mode
        self.feat = feat
        
    def __len__(self):
        return self.fname.shape[0]
    
    def __getitem__(self, idx):
        filename = self.fname['id'].iloc[idx]
        age = self.fname['age'].iloc[idx]
        age = (age - 39.244260331403474)/20.44699709099905
        gender = self.fname['gender'].iloc[idx]
        if self.mode == 'train':
            filepath = os.path.join('hf_round2_train/', filename.split('.')[0]+'.txt')
        else:
            filepath = os.path.join('../tcdata/hf_round2_testB/', filename.split('.')[0]+'.txt')
        data = pd.read_csv(filepath, sep=' ')
        data['III'] = data['II']-data['I']
        data['aVR']=-(data['II']+data['I'])/2
        data['aVL']=(data['I']-data['II'])/2
        data['aVF']=(data['II']-data['I'])/2
        data = data.astype('float').values
        data = resample(data, 2500).transpose()
        fea = self.feat[idx].reshape(1,-1)
        fea2 = np.zeros((4,))
        fea2[0] = age
        if gender == 'MALE':
            fea2[1] = 1
        if gender == 'FEMALE':
            fea2[2] = 1
        if gender == 'MISS':
            fea2[3] = 1
        if self.transform is not None:
            data = self.transform(data)
            fea = self.transform(fea)
            fea2 = self.transform(fea2)
        return data, fea,fea2, filename
    
class Model(nn.Module):
    def __init__(self, base):
        super(Model, self).__init__()
        self.base = base
        self.bn1 = nn.BatchNorm1d(1)
        self.classifier = nn.Linear(1024+112, 34)
    def forward(self, x1, x2, x3):
        x1 = self.base(x1)
        x2 = self.bn1(x2)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat([x1,x2,x3], 1)
        x = self.classifier(x)
        return x
    
model = Model(densenet121()).cuda()
testres = []
for foldNum in range(5):
    testData = TestData(subA, 'testA', Feat_RR, transform=transforms.Compose([ToTensor()]))
    testloader = DataLoader(testData, batch_size=64, shuffle=False)
    model.load_state_dict(torch.load('dense121/weight_best_%s.pt'% str(foldNum)))
    model.cuda()
    model.eval()
    test_outputs = []
    test_fnames = []
    for images, fea, fea2, fnames in testloader:
        preds = torch.sigmoid(model(images.cuda(), fea.cuda(), fea2.cuda()).detach())
        test_outputs.append(preds.cpu().numpy())
        test_fnames.extend(fnames)
        
    test_preds = pd.DataFrame(data=np.concatenate(test_outputs),
                                  index=test_fnames,
                                  columns=map(str, range(34)))
    test_preds = test_preds.groupby(level=0).mean()
    testres.append(test_preds)
    print(2)
test2_dense121 = (testres[0]+testres[1]+testres[2]+testres[3]+testres[4])/5

class Model(nn.Module):
    def __init__(self, base):
        super(Model, self).__init__()
        self.base = base
        self.bn1 = nn.BatchNorm1d(1)
        self.classifier = nn.Linear(624, 34)
    def forward(self, x1, x2, x3):
        x1 = self.base(x1)
        x2 = self.bn1(x2)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat([x1,x2,x3], 1)
        x = self.classifier(x)
        return x

model = Model(resnet34()).cuda()
testres = []
for foldNum in range(5):
    testData = TestData(subA, 'testA', Feat_RR, transform=transforms.Compose([ToTensor()]))
    testloader = DataLoader(testData, batch_size=64, shuffle=False)
    model.load_state_dict(torch.load('resnet34/weight_best_%s.pt'% str(foldNum)))
    model.cuda()
    model.eval()
    test_outputs = []
    test_fnames = []
    for images, fea, fea2, fnames in testloader:
        preds = torch.sigmoid(model(images.cuda(), fea.cuda(), fea2.cuda()).detach())
        test_outputs.append(preds.cpu().numpy())
        test_fnames.extend(fnames)
        
    test_preds = pd.DataFrame(data=np.concatenate(test_outputs),
                                  index=test_fnames,
                                  columns=map(str, range(34)))
    test_preds = test_preds.groupby(level=0).mean()
    testres.append(test_preds)
    print(2)
test2_resnet34 = (testres[0]+testres[1]+testres[2]+testres[3]+testres[4])/5

class Model(nn.Module):
    def __init__(self, base):
        super(Model, self).__init__()
        self.base = base
        self.bn1 = nn.BatchNorm1d(1)
        self.classifier = nn.Linear(2160, 34)
    def forward(self, x1, x2, x3):
        x1 = self.base(x1)
        x2 = self.bn1(x2)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat([x1,x2,x3], 1)
        x = self.classifier(x)
        return x

model = Model(se_resnet101(num_classes=34, pretrained=None)).cuda()
testres = []
for foldNum in range(5):
    testData = TestData(subA, 'testA', Feat_RR, transform=transforms.Compose([ToTensor()]))
    testloader = DataLoader(testData, batch_size=64, shuffle=False)
    model.load_state_dict(torch.load('se_resnet101/weight_best_%s.pt'% str(foldNum)))
    model.cuda()
    model.eval()
    test_outputs = []
    test_fnames = []
    for images, fea, fea2, fnames in testloader:
        preds = torch.sigmoid(model(images.cuda(), fea.cuda(), fea2.cuda()).detach())
        test_outputs.append(preds.cpu().numpy())
        test_fnames.extend(fnames)
        
    test_preds = pd.DataFrame(data=np.concatenate(test_outputs),
                                  index=test_fnames,
                                  columns=map(str, range(34)))
    test_preds = test_preds.groupby(level=0).mean()
    testres.append(test_preds)
    print(2)
test2_se_resnet101 = (testres[0]+testres[1]+testres[2]+testres[3]+testres[4])/5

model = Model(se_resnet50(num_classes=34, pretrained=None)).cuda()
testres = []
for foldNum in range(5):
    testData = TestData(subA, 'testA', Feat_RR, transform=transforms.Compose([ToTensor()]))
    testloader = DataLoader(testData, batch_size=64, shuffle=False)
    model.load_state_dict(torch.load('se_resnet50/weight_best_%s.pt'% str(foldNum)))
    model.cuda()
    model.eval()
    test_outputs = []
    test_fnames = []
    for images, fea, fea2, fnames in testloader:
        preds = torch.sigmoid(model(images.cuda(), fea.cuda(), fea2.cuda()).detach())
        test_outputs.append(preds.cpu().numpy())
        test_fnames.extend(fnames)
        
    test_preds = pd.DataFrame(data=np.concatenate(test_outputs),
                                  index=test_fnames,
                                  columns=map(str, range(34)))
    test_preds = test_preds.groupby(level=0).mean()
    testres.append(test_preds)
    print(2)
test2_se_resnet50 = (testres[0]+testres[1]+testres[2]+testres[3]+testres[4])/5

label_map_reverse = {v:k for (k,v) in label_map.items()}
with open('../tcdata/hf_round2_subB.txt', 'r') as f:
    su = f.readlines()
    su = [x[:-1] for x in su]
map2 = {0: 0.13,
 1: 0.45,
 2: 0.67,
 3: 0.47000000000000003,
 4: 0.46,
 5: 0.44,
 6: 0.46,
 7: 0.33,
 8: 0.45,
 9: 0.45,
 10: 0.5,
 11: 0.2,
 12: 0.43,
 13: 0.19,
 14: 0.71,
 15: 0.39,
 16: 0.43,
 17: 0.4,
 18: 0.48,
 19: 0.4,
 20: 0.43,
 21: 0.3,
 22: 0.53,
 23: 0.24,
 24: 0.54,
 25: 0.43,
 26: 0.44,
 27: 0.35000000000000003,
 28: 0.43,
 29: 0.42,
 30: 0.58,
 31: 0.55,
 32: 0.43,
 33: 0.53}
test2 = (test2_resnet34*0.3 + test2_dense121*0.2 + test2_se_resnet101*0.3  + test2_se_resnet50*0.2 )
for i in range(34):
    test2[str(i)] = (test2[str(i)] > map2[i]).astype(int)
#test2 = (test2 > 0.5).astype(int)
test2 = test2.reset_index().rename(columns={'index':'id'})
kkk2 = pd.merge(subA, test2, on='id', how='left')
with open('result.txt', 'w',encoding='utf-8') as f:
    for i in range(kkk2.shape[0]):
        data = kkk2.iloc[i]
        res = su[i]
        for j in range(34):
            if data[str(j)] == 1:
                res += '\t'
                res += label_map_reverse[j]
        res += '\n'
        f.write(res)
