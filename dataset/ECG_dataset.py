# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
# import 
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import os
import numpy as np
import scipy.signal as sig
import numpy as np
import pandas as pd
#b,c,h,w
#b,c d
# import h5py
import torch # to set manual seed
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wfdb # wfdb?
from scipy.signal import stft
import os
import sys
import numpy as np
from scipy.ndimage import zoom
from scipy import signal as signal_fc
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath('/home/mimic/MIMIC_subset/MIMIC_subset'))
# from dataset.ECG_dataset import get_ECG_datasets,get_data_loader

# print(a)
from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()

def adjust_sig(signal):

    new_sig = sig.resample(signal, 4000, axis=0)
    #TODO: change （1，1）to（48，48）
    new_sig = np.pad(new_sig, ((48,48),(0,0)), mode='constant')
    new_sig = new_sig.astype(np.float32) # store as float32, else OOM
    return new_sig
    
def generate_file_path(base_path, split):
    # 使用 f-string 生成路径
    return f"{base_path}{split}.csv"

#=====高通滤波器===========
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal_fc.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpassfilter(data, cutoff=1, fs=400, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal_fc.filtfilt(b, a, data)
    return y

#使用样例
#input e_signal的size为[12,4096]
# filtered_signal=highpassfilter(e_signal)
#filtered_signal的size是[12,4096]
#======高通滤波器==========



class MIMICECG(Dataset):
    def __init__(self,  fullinfo_file,transform=None, split='train',module='fusion'):#delete args in params

        self.args = args

        self.split=split
        self.module = module
        self.fullinfo_path=fullinfo_file
        file_path = generate_file_path(fullinfo_file, self.split)

        # self.ecg_dir='/home/mimic/MIMIC_subset/MIMIC_subset/ECG_data'
        self.ecg_dir='/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ecg/1.0'
        with open(file_path, "r")as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]

        # self.CLASSES=self._listfile_header.strip().split(',')[6:13]
        self._data = self._data[1:]

        self._data = [line.split(',') for line in self._data]


        self.data_map = {}

        # self.data_map = {
        #     mas[1]: {
        #         #  'labels': [1 if label == '1' else 0 for label in mas[9:16]],
        #         'labels': [1 if mas[i] == '1.0' else 0 for i in [6, 7, 8, 9,10, 11,12]],

        #         'hadm_id': float(mas[1]),
        #         'first_ecg_path': mas[4],
        #         # 'dicom_id': float(mas[14]),
        #         }
        #         for mas in self._data
        # }
        self.data_map = {
            mas[1]: {
                # 'labels': [1 if label == '1' else 0 for label in mas[10,11,12,13]],
                'labels': [1 if mas[i] == '1' else 0 for i in [10, 11,12,13]],

                'hadm_id': float(mas[1]),
                # 'hadm_id': float(mas[1]),
                'ecg_path': mas[5].strip(),
                }
                for mas in self._data
        }

        self.data_map = {hadm_id: data for hadm_id, data in self.data_map.items() if any(data['labels'])}

        self.names = list(self.data_map.keys())

        self.transform = transform

    def __getitem__(self, index):
        # print(f'get_item index {index}')
        if isinstance(index, int):
            index1 = self.names[index]
            # print(index)#is hadm_id
            y = self.data_map[index1]['labels']

            # print(y)
            study_whole_path = self.data_map[index1]['ecg_path']
            # study_id_path='/'.join(study_whole_path.split('/')[-2:])
            rec_path=f'{self.ecg_dir}/{study_whole_path}'
            # print(f'get item path {rec_path}')
            rd_record = wfdb.rdrecord(rec_path)

            signal = rd_record.p_signal # 5k x 12 signal  #not none

            signal=adjust_sig(signal)#4096,12
            # for i in range(signal.shape[1]):
            #     # tmp = pd.DataFrame(signal[:,i]).interpolate().values.ravel().tolist()
            #     signal[:,i]= np.clip(signal[:,i],a_max=5, a_min=-5) 
            
            if np.isnan(signal).any():
                print("Data contains NaN values. Dropping the entire data.")
                # 清空数据（或其他处理逻辑）
                signal = None  

            #添加高通滤波
            signal=signal.transpose(1,0)
            signal=highpassfilter(signal)
            signal=signal.transpose(1,0)
            


            if signal is not None:
                signal = torch.tensor(signal, dtype=torch.float32)
                signal=signal.permute(1,0)


# ============先注释掉==============================
                if self.args.domain=='frequency':
                    print('------------couculate frequency mean,std--------------------------')
                    print(f'0 {signal.shape}') #[12,4096]
                    f,t, Zxx = stft(signal,fs=400, window='hann',nperseg=512,noverlap=256) #signal=[channel,dim]
                    signal = np.abs(Zxx)
                    print(f'stft signal {signal.shape}')#[12,257,17]
                    # print(f'signal {signal.shape}')#[12,257,17]
                    if self.args.domain=='frequency' and self.args.fusion_type=='ecg' and self.args.ecg_model=='resnet18':
                        scale_factor = (1, 224 / 257, 224 / 17)  # (通道数不变, 高度缩放因子, 宽度缩放因子)

    # 使用 zoom 进行插值调整
                        signal = zoom(signal, scale_factor)

                    signal = signal.transpose(1,2,0)
                    print(f'frequency initial signal{signal}')

                    # if self.split=='train':
                    #     print(f'get item signal {signal.shape}')#[257,17,12]
                    #     # normalize / must have channel last 

                    #     mean=torch.tensor([0.03296802565455437, 0.03273816406726837, 0.03355516120791435, 0.0311842393130064, 0.030348533764481544, 0.027099210768938065, 0.024589119479060173, 0.023644540458917618, 0.02395729161798954, 0.023933202028274536, 0.0230211541056633, 0.021562350913882256])
                    #     std=torch.tensor([0.04679363593459129, 0.0376615896821022, 0.03844350948929787, 0.0354924313724041, 0.03010416217148304, 0.024544646963477135, 0.022233400493860245, 0.021855955943465233, 0.020977528765797615, 0.0198527779430151, 0.018527885898947716, 0.017392240464687347])

                    #     signal = torch.tensor(signal)

                    #     signal=(signal-mean) / std
                    #     signal = signal.permute(2, 0, 1)

                        

                    # if self.split=='val':

                    #     signal = torch.tensor(signal)
                    #     mean=torch.tensor([0.033378660678863525, 0.03305447846651077, 0.034193579107522964, 0.031550049781799316, 0.03091508150100708, 0.02711910754442215, 0.024816736578941345, 0.02398015186190605, 0.02438199333846569, 0.02432316541671753, 0.023364150896668434, 0.021768491715192795])
                    #     std=torch.tensor([0.04679363593459129, 0.0376615896821022, 0.03844350948929787, 0.0354924313724041, 0.03010416217148304, 0.024544646963477135, 0.022233400493860245, 0.021855955943465233, 0.020977528765797615, 0.0198527779430151, 0.018527885898947716, 0.017392240464687347])

                    #     signal=(signal-mean) / std
                    #     signal = signal.permute(2, 0, 1)

                    # if self.args.domain=='frequency' and self.args.fusion_type=='ecg' and self.args.ecg_model=='resnet18':

                    #     signal = signal.numpy()


                

                else:
    #normalize
                    if self.split=='train':
                        print(f'--------norm of fusion dataset-----------')

                        mean=torch.tensor([0.021983036771416664, 0.013289721682667732, -0.007492052856832743, -0.0172954760491848, 0.002315341029316187, 0.014129746705293655, -0.018112635239958763, -0.0034318040125072002, 0.00021024956367909908, 0.014880170114338398, 0.02024325355887413, 0.01977406069636345])
                        std=torch.tensor([0.15690241754055023, 0.15237219631671906, 0.15942928194999695, 0.13163667917251587, 0.13473273813724518, 0.13911446928977966, 0.18939709663391113, 0.25870004296302795, 0.2708594501018524, 0.2404349446296692, 0.21415464580059052, 0.18362797796726227])

                        mean=mean.view(-1,1)

                        signal=(signal-mean.view(-1,1)) / std.view(-1,1)

                    if self.split=='val':

                        mean=torch.tensor([0.02482433430850506, 0.015947148203849792, -0.007936287671327591, -0.019916854798793793, 0.0034988929983228445, 0.01582620106637478, -0.019720222800970078, -0.004102816805243492, -0.0021764931734651327, 0.012111539952456951, 0.021004941314458847, 0.02135436423122883])
                        std=torch.tensor([0.1607745736837387, 0.15428286790847778, 0.16370441019535065, 0.1342308074235916, 0.13679346442222595, 0.14322689175605774, 0.1934480518102646, 0.2608761787414551, 0.27516594529151917, 0.24000638723373413, 0.2119644731283188, 0.18333952128887177])
                        signal=(signal-mean.view(-1,1)) / std.view(-1,1)
    #==========================先注释                   




                if self.transform is not None:
                    # signal = torch.tensor(signal, dtype=torch.float32)
                    # print(f'signal {signal.shape}')
                    signal = self.transform(signal)

                return index1,signal,y

    def __len__(self):
        return len(self.data_map)
    

def get_transforms():#delete rgs param

    train_transforms = []


    test_transforms = []


    return train_transforms, test_transforms


def get_ECG_datasets(args,module='unimodal'):#delete args in params

    train_transforms, test_transforms = get_transforms()#delete args param

    fullinfo_file='/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_'

    dataset_train = MIMICECG(fullinfo_file, split='train',module=module)#delete args in params
    # print(f'train')
    dataset_validate = MIMICECG(fullinfo_file, split='val',module=module)#delete args in params
    # print(f'val')
    dataset_test = MIMICECG(fullinfo_file, split='test',module=module)#delete args in params
    # print(f'test')
    return dataset_train, dataset_validate, dataset_test


def get_data_loader(batch_size):
    train_ds, val_ds, test_ds = get_ECG_datasets(args)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16)

    return train_dl, val_dl

def my_collate(batch):
    # 将样本的输入数据（x）堆叠成一个批次
    #TODO: initial:
    batch = [item for item in batch if item is not None]

    index1=np.stack([item[0] for item in batch])
    x = np.stack([item[1] for item in batch])


    targets = np.array([item[2] for item in batch])
    
    # 返回输入数据和对应的目标标签
    return [index1,x, targets]

# train_ecg_dl, val_ecg_dl = get_data_loader(batch_size=args.batch_size)

# for i,data in enumerate(train_ecg_dl):
#     print(i)


