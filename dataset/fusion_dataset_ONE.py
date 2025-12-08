
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
np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
torch.set_printoptions(profile="full", linewidth=200)
import h5py
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
# change it to your local path
sys.path.append(os.path.abspath('/home/mimic/MIMIC_subset/MIMIC_subset'))
# from dataset.ECG_dataset import get_ECG_datasets,get_data_loader

# print(a)
from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
# from dataset.ECG_dataset import get_ECG_datasets,get_data_loader

# mport torch 
# torch.set_printoptions(threshold=np.inf)

# Adjust ECGs to match Code-15 standard
def adjust_sig(signal):
    # MIMIC IV signals are 10s at 500Hz, 5k total samples
    # we seek 10s @ 400Hz, centered, padded to 4096.
    # so resample to 4k samples
    # then pad with 48 0's on both sides of axis=0
    new_sig = sig.resample(signal, 4000, axis=0)
    #TODO: change （1，1）to（48，48）
    new_sig = np.pad(new_sig, ((48,48),(0,0)), mode='constant')
    new_sig = new_sig.astype(np.float32) # store as float32, else OOM
    return new_sig
    
def generate_file_path(base_path, split):
    # 使用 f-string 生成路径
    return f"{base_path}{split}.csv"

def generate_file_path(base_path, split):
    # 使用 f-string 生成路径
    return f"{base_path}{split}.csv"

def get_image_path(cxr_rootpath, dicom_id):
    # 构造完整的图像路径
    image_filename = f"{dicom_id}.jpg"
    image_path = os.path.join(cxr_rootpath, image_filename)
    return image_path

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal_fc.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpassfilter(data, cutoff=1, fs=400, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal_fc.filtfilt(b, a, data)
    return y


class MIMIC_ECG_CXR(Dataset):
    def __init__(self,  fullinfo_file,transform_i=None,transform_e=None, split='train'):#delete args in params
        # self.data_dir = args.cxr_data_dir
        # cxr_data_dir='/data/ke/data/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0'
        # fullinfo_file='/data/ke/MIMIC_subset/PA_subset/with_nonan_label_PA_'
        self.args = args
    #     self.CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    #    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    #    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    #    'Pneumonia', 'Pneumothorax', 'Support Devices']\
        self.split=split
        # self.module = module
        self.fullinfo_path=fullinfo_file
        file_path = generate_file_path(fullinfo_file, self.split)
#change to the path where cxr has been resized
        self.cxr_rootpath='/mnt/oldvolume1/home/mimic/MIMIC_subset/MIMIC_subset/resized'
#change to the ecg downloaded path
        self.ecg_dir='/mnt/oldvolume1/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ecg/1.0'

        # self.filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}
        # self.base_path='/data/ke/MIMIC_subset/PA_subset/with_nonan_label_PA_'
        with open(file_path, "r")as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES=self._listfile_header.strip().split(',')[6:13]
        self._data = self._data[1:]

        self._data = [line.split(',') for line in self._data]

        # label_indices = {label: idx + 6 for idx, label in enumerate(self.CLASSES)}

        self.data_map = {}

        self.data_map = {
            mas[1]: {
                # 'labels': [1 if label == '1' else 0 for label in mas[10,11,12,13]],
                # 'labels': [1 if mas[i] == '1' else 0 for i in [21,22,23]],

                'labels': [1 if mas[i] == '1' else 0 for i in [21,22,23]],  # 结果为 [1, 1, 1]
                'hadm_id': float(mas[1]),
                # 'hadm_id': float(mas[1]),
                'dicom_id': mas[7].strip(),
                'ecg_path': mas[6].strip(),
                'clic_value' : [
                self._fill_default(mas[12], 0),  # '51003'
                self._fill_default(mas[13], 1.4),   # '50908'
                self._fill_default(mas[14], 0.6),   # '50963'
                self._fill_default(mas[15], 0),  # 'temperature'
                self._fill_default(mas[16], 0),  # 'heartrate'
                self._fill_default(mas[17], 0),  # 'resprate'
                self._fill_default(mas[18], 0),  # 'o2sat'
                self._fill_default(mas[19], 0), # 'sbp'
                self._fill_default(mas[20], 0),  # 'dbp'
                 ]
                }

                for mas in self._data
        }
        # for mas in self._data:
        #     clic_value = [
        #         self._fill_default(mas[15], 0.05),  # '51003'
        #         self._fill_default(mas[16], 3.0),   # '50908'
        #         self._fill_default(mas[17], 0.5),   # '50963'
        #         self._fill_default(mas[18], 98.0),  # 'temperature'
        #         self._fill_default(mas[19], 80.0),  # 'heartrate'
        #         self._fill_default(mas[20], 16.0),  # 'resprate'
        #         self._fill_default(mas[21], 98.0),  # 'o2sat'
        #         self._fill_default(mas[22], 110.0), # 'sbp'
        #         self._fill_default(mas[23], 70.0),  # 'dbp'
        #     ]


        # print(len(self.data_map))#train 10632;test/val:1653
        self.names = list(self.data_map.keys())
        # print(self.data_map)
        self.transform_e = transform_e
        self.transform_i=transform_i

    def normalize(self,value, min_val, max_val):
       
        if max_val == min_val:  
            return 0
        return (value - min_val) / (max_val - min_val)

    def _fill_default(self, value, default_value):
        """Helper function to return the default value if the provided value is invalid."""
        try:
            # Try converting the value to float
            return float(value) if value not in [None, ''] else default_value
        except ValueError:
            # If the value cannot be converted, return the default
            return default_value




    def __getitem__(self, index):
        # print(f'get_item index {index}')
        if isinstance(index, int):
            index1 = self.names[index]
            # print(index)#is hadm_id
            y = self.data_map[index1]['labels']
            # print(y)
            study_whole_path = self.data_map[index1]['ecg_path']
            # print(f'study_whole_path {study_whole_path}')
            # study_id_path='/'.join(study_whole_path.split('/')[-2:])
            dicom_id = self.data_map[index1]['dicom_id']
            clic_value = self.data_map[index1]['clic_value']
            # print(f'clic_value {clic_value}')
            normalization_params = {
            0: (0, 0.01),  # '51003'
            1: (0, 6),     # '50908'
            2: (0, 6),     # '50963'
            3: (97.8, 99), # 'temperature'
            4: (60, 100),  # 'heartrate'
            5: (12, 16),   # 'resprate'
            6: (95, 100),  # 'o2sat'
            7: (90, 120),  # 'sbp'
            8: (60, 80)    # 'dbp'
        }
        
        # 归一化 clic_value
            normalized_clic_value = [
                self.normalize(clic_value[i], normalization_params[i][0], normalization_params[i][1]) 
                for i in range(len(clic_value))
            ]
            # print(f'normalized_clic_value {normalized_clic_value}')

            img_path= get_image_path(self.cxr_rootpath, dicom_id)
            # print(f'get cxr path {img_path}')
            rec_path=f'{self.ecg_dir}/{study_whole_path}'
            # print(f'get ecg path {rec_path}')
            rd_record = wfdb.rdrecord(rec_path)
            

            signal = rd_record.p_signal # 5k x 12 signal  #not none


            # if np.isnan(signal).any():
            #     print(f'signal contain nan')
            #     return None  

            signal=adjust_sig(signal)#4096,12
            # print(f'signal {signal}')
            # print("signal (first 100 time steps):")
            # print(signal[:, :100])
            # for i in range(signal.shape[1]):
            #     # tmp = pd.DataFrame(signal[:,i]).interpolate().values.ravel().tolist()
            #     signal[:,i]= np.clip(signal[:,i],a_max=3, a_min=-3) 
            
            if np.isnan(signal).any():
                # print("Data contains NaN values. Dropping the entire data.")
               
                signal = None  

            if signal is not None:

                signal = torch.tensor(signal, dtype=torch.float32)
                signal=signal.permute(1,0)
                signal=highpassfilter(signal)
                # print(f'after filter {signal.shape}')#[12,4096]

                if self.args.domain=='frequency' :

                    if self.args.fusion_type=='fre_FRSU':

                        # signal=signal
                        # signal=signal.transpose(1,0)

                        f,t, Zxx = stft(signal,fs=100, window='hann',nperseg=512,noverlap=256) #signal=[channel,dim]

                        signal = np.abs(Zxx) #[B,channel,257,17]
                        signal = signal.transpose(1,2,0)


                    else:
                        signal=signal
                        signal=signal.transpose(1,0)

            

            if not os.path.exists(img_path):
                print(f'miss ID: {index1}, path: {img_path}')  
                return None  


            img = Image.open(img_path).convert('RGB')






            if self.transform_e is not None:
                # signal = torch.tensor(signal, dtype=torch.float32)
                signal = self.transform_e(signal)
                
            if self.transform_i is not None:
                # signal = torch.tensor(signal, dtype=torch.float32)
                img = self.transform_i(img)
                

            # if self.module=='fusion':

            #     return {index:(signal, y)}
            # else:
                # print(f'model!=fusion signal {signal}')
                # print(f'before return signal {signal.shape}')
                return signal,img,normalized_clic_value,y



    def __len__(self):
        return len(self.data_map)
    

def get_transforms():#delete rgs param
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transforms = []
    train_transforms.append(transforms.Resize(256))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    train_transforms.append(transforms.CenterCrop(224))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(normalize)      


    test_transforms = []
    test_transforms.append(transforms.Resize(256))


    test_transforms.append(transforms.CenterCrop(224))

    test_transforms.append(transforms.ToTensor())
    test_transforms.append(normalize)


    return train_transforms, test_transforms


def load_cxr_ecg_ds():
    train_transforms, test_transforms = get_transforms()#delete args param

    cxr_rootpath='/mnt/oldvolume1/home/mimic/MIMIC_subset/MIMIC_subset/resized'
    #change it to the path of extracted list(PA_subset_new in OSC)
    fullinfo_file='/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/PA_subset/with_nonan_label_PA_'

    dataset_train = MIMIC_ECG_CXR(fullinfo_file,  split='train', transform_i=transforms.Compose(train_transforms))#delete rgs param
    dataset_validate = MIMIC_ECG_CXR(fullinfo_file,  split='val', transform_i=transforms.Compose(test_transforms))#delete rgs param
    # dataset_test = MIMIC_ECG_CXR(fullinfo_file, split='test', transform_i=transforms.Compose(test_transforms))#delete rgs param

    return dataset_train, dataset_validate


from torch.utils.data import Subset, DataLoader


subset_indices = list(range(10))
# train_ds = Subset(train_ds, subset_indices)



def get_ecgcxr_data_loader(batch_size):
    train_ds, val_ds = load_cxr_ecg_ds()

    # train_ds = Subset(train_ds, subset_indices)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16,drop_last=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16,drop_last=True)

    return train_dl, val_dl

def my_collate(batch):
    # batch = [item for item in batch if item is not None]
    batch = [item for item in batch if item is not None and item[0] is not None]  # 过滤掉 item[0] 为 None 的项


    # for item in batch:
    #     print(f'item {item}')
    # print(f'collection {len(batch)}')
    ecg_data = [item[0] for item in batch]
    # pairs = [False if item[1] is None else True for item in batch]
    pairs = [False if item[1] is None else True for item in batch]

    img = torch.stack([item[1] for item in batch])
    clic_value = torch.stack([torch.tensor(item[2]) for item in batch])

    # clic_value = torch.stack([item[2] for item in batch])
    # x, seq_length = pad_zeros(x)

    seq_length = [x.shape[0] for x in ecg_data]

    # max_len = max(seq_length)
    targets = np.array([item[3] for item in batch])
    # targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
    return [ecg_data,img,clic_value,targets,seq_length,pairs]

