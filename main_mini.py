
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import random
print("CWD:", os.getcwd())
seed = 42
random.seed(seed)  
np.random.seed(seed)  
torch.manual_seed(seed)  
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

# from dataset.fusion_dataset_ONE_10fold import load_cxr_ecg_ds,get_ecgcxr_data_loader
from dataset.fusion_dataset_pul import load_cxr_ecg_ds_pul,get_ecgcxr_data_loader_pul
from dataset.fusion_dataset_ONE import load_cxr_ecg_ds,get_ecgcxr_data_loader

import numpy as np
from train.train_unimodal import uni_trainer
from train.train_hyperbolic_org import bolic_trainer 
 

from model.hpblic_vgg19 import DDMF_Net_hpblic7_2
from model.hpblic_ENet import DDMF_Net_hpblic_ENet
from model.hpblic_trans import DDMF_Net_hpblic_trans
from model.unimodal_cxr import uni_cxr
from model.finial_model import DDMF_Net_hpblic7_2





from argument import args_parser
parser = args_parser()

args = parser.parse_args()


torch.manual_seed(seed)  
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False



fusion_train_dl,fusion_val_dl=get_ecgcxr_data_loader(batch_size=args.batch_size)
fusion_train_dl_pul,fusion_val_dl_pul=get_ecgcxr_data_loader_pul(batch_size=args.batch_size)


if args.fusion_model=='uni_cxr':
    fusion_model=uni_cxr()

elif args.fusion_model=='hpblic':
    fusion_model=DDMF_Net_hpblic7_2()



elif args.fusion_type=='hpblic' :
    print(f'--------start fusion training-------------')
    trainer=bolic_10fold_trainer(fusion_train_dl, fusion_val_dl,args,fusion_model)
    trainer.train()

