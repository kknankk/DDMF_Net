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

def generate_file_path(base_path, split):
    # 使用 f-string 生成路径
    return f"{base_path}{split}.csv"

def get_image_path(cxr_rootpath, dicom_id):
    # 构造完整的图像路径
    image_filename = f"{dicom_id}.jpg"
    image_path = os.path.join(cxr_rootpath, image_filename)
    return image_path

class MIMICCXR(Dataset):
    def __init__(self, fullinfo_file,cxr_rootpath,transform=None, split='train',module='fusion'):#delete args in params
        # self.data_dir = args.cxr_data_dir
        # cxr_data_dir='/data/ke/data/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0'
        fullinfo_file='/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_'
        # self.args = args
    #     self.CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    #    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    #    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    #    'Pneumonia', 'Pneumothorax', 'Support Devices']
        # self.fullinfo_path=fullinfo_file
        file_path = generate_file_path(fullinfo_file, split)
        self.module = module
        # self.filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}
        # self.cxr_rootpath='/data/ke/data/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/resized'
        self.cxr_rootpath='/home/mimic/MIMIC_subset/MIMIC_subset/resized'
        # self.cxr_rootpath='/home/ke/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_'
        
        with open(file_path, "r")as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        # self.CLASSES=self._listfile_header.strip().split(',')[6:13]
        self._data = self._data[1:]
        self._data = [line.split(',') for line in self._data]

        # metadata_with_labels = metadata.merge(labels[self.CLASSES+['study_id'] ], how='inner', on='study_id')

        
        # self._data = [line.split(',') for line in self._data]
        self.data_map = {
            mas[1]: {
                # 'labels': [1 if label == '1' else 0 for label in mas[10,11,12,13]],
                'labels': [1 if mas[i] == '1' else 0 for i in [10, 11,12,13]],

                'hadm_id': float(mas[1]),
                # 'hadm_id': float(mas[1]),
                'dicom_id': mas[6].strip(),
                }
                for mas in self._data
        }
        self.names = list(self.data_map.keys())


        #  y = self.data_map[index]['labels']
        # self.filesnames_to_labels = dict(zip(metadata_with_labels['dicom_id'].values, metadata_with_labels[self.CLASSES].values))
        # self.filenames_loaded = splits.loc[splits.split==split]['dicom_id'].values
        
        self.transform = transform
        # self.filenames_loaded = [filename  for filename in self.filenames_loaded if filename in self.filesnames_to_labels]
       

#TODO: confirm: dicom_id = self.data_map[index]['dicom_id']
    def __getitem__(self, index):
        if isinstance(index, int):
            index1 = self.names[index]
            y = self.data_map[index1]['labels']
            # print(f'index {index}')
            # print(f'cxr label {y}')
    

            dicom_id = self.data_map[index1]['dicom_id']
            img_path= get_image_path(self.cxr_rootpath, dicom_id)
            # print(f'img path {img_path}')
            # print(img_path)

            if not os.path.exists(img_path):
                # print(f'缺失文件 ID: {index1}, 路径: {img_path}')  # 打印缺失文件的 ID 和路径
                return None  # 可以返回 None 或者其他默认值
       




            
            img = Image.open(img_path).convert('RGB')
            # img_array = np.array(img)  # 将 PIL 图像转换为 NumPy 数组
            # print("After convert ro RGB, size:", img_array.shape)


            # labels = torch.tensor(self.filesnames_to_labels[index]).float()
            # self.filenames_loaded = self.data_map[index]['labels']

            if self.transform is not None:
                img = self.transform(img)
            # return img, labels
            if self.module=='fusion':
                print(f'module=fusion')

                return {index:img}
            else:
                return index,img,y

        
        # filename = self.filenames_loaded[index]
        
        # img = Image.open(self.filenames_to_path[dicom_id]).convert('RGB')

        # labels = torch.tensor(self.filesnames_to_labels[filename]).float()

        # if self.transform is not None:
        #     img = self.transform(img)
        # # return img, labels
        # return {index:img}
    
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

def get_cxr_datasets(module='unimodal'):#delete args param
    train_transforms, test_transforms = get_transforms()#delete args param

    cxr_rootpath='/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC_CXRdataset'
    fullinfo_file='/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_'

    # filepath = f'{args.cxr_data_dir}/paths.npy'
    # if os.path.exists(filepath):
    #     paths = np.load(filepath)
    # else:
    # paths = glob.glob(f'{data_dir}/resized/**/*.jpg', recursive = True)
    # np.save(filepath, paths)
    
    dataset_train = MIMICCXR(fullinfo_file,cxr_rootpath,  split='train', transform=transforms.Compose(train_transforms),module=module)#delete rgs param
    dataset_validate = MIMICCXR(fullinfo_file,cxr_rootpath,  split='val', transform=transforms.Compose(test_transforms),module=module)#delete rgs param
    # dataset_test = MIMICCXR(fullinfo_file,cxr_rootpath, split='test', transform=transforms.Compose(test_transforms),module=module)#delete rgs param

    return dataset_train, dataset_validate

# A,B,C=get_cxr_datasets()

# image = A.__getitem__(index=5)

def get_cxrdata_loader(batch_size):
    train_ds,val_ds=get_cxr_datasets()
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)
    # test_dl = DataLoader(test_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)

    return train_dl, val_dl


def my_collate(batch):
    # 将样本的输入数据（x）堆叠成一个批次
    batch = [item for item in batch if item is not None]
    index = np.stack([item[0] for item in batch])
    x = np.stack([item[1] for item in batch])
    targets = np.array([item[2] for item in batch])

    
    # 将目标标签（y）转换为NumPy数组
    # targets = np.array([item[1] for item in batch])
    
    # 返回输入数据和对应的目标标签
    return [index,x,targets]


# train_ds,val_ds,test_ds=get_cxr_datasets()
# train_dl,val_dl,test_dl=get_cxrdata_loader(batch_size=4)

# for i,x in enumerate(train_ds):
#     print(i)
    # print(f'train_ds img {x}')
