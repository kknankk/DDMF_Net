# #!/usr/bin/python

# import thread
import time
from PIL import Image
import glob
from tqdm import tqdm
import os

print('starting')
# data_dir = '/data/ke/data/MIMIC/physionet.org/files/mimic-cxr-jpg'
# version = '2.0.0'
# data_dir = '/home/mimic/MIMIC_subset/MIMIC_subset/download_xcr'
# version = '2.0.0'
data_dir ='/home/ec2-user/yuan/dataset/multimodal/mimic/CXR_data/physionet.org/files/mimic-cxr-jpg/2.0.0/files'

# paths_done = glob.glob(f'{data_dir}/{version}/resized/**/*.jpg', recursive = True)
# print('done', len(paths_done))
paths_done = glob.glob(f'/home/ec2-user/yuan/dataset/multimodal/mimic/resized/*.jpg', recursive = True)
print('done', len(paths_done))

# paths_all = glob.glob(f'{data_dir}/*.jpg', recursive = True)
# # print(paths_all)#精确到jpg路径
# print('all', len(paths_all))


paths_all = glob.glob(f'{data_dir}/**/*.jpg', recursive = True)
print('all', len(paths_all))

done_files = [os.path.basename(path) for path in paths_done]
# print(done_files)
paths = [path for path in paths_all if os.path.basename(path) not in done_files ]
print('left', len(paths))

def resize_images(path):
    try:
        basewidth = 512
        filename = path.split('/')[-1]
        img = Image.open(path)

        wpercent = (basewidth/float(img.size[0]))
        
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize))
        print(f'')
        
        img.save(f'/home/ec2-user/yuan/dataset/multimodal/mimic/resized/{filename}')

    except OSError as e:
        print(f"Error processing file {path}: {e}")



from multiprocessing.dummy import Pool as ThreadPool

threads = 10

for i in tqdm(range(0, len(paths), threads)):
    paths_subset = paths[i: i+threads]
    pool = ThreadPool(len(paths_subset))
    pool.map(resize_images, paths_subset)
    pool.close()
    pool.join()
