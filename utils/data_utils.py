#from .twoline_dataset import *
#from .transformations import *



import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize

def print_info(arr):
    print(f"{arr.shape=}  {arr.max()=}   {arr.min()=}  {arr.mean()=} {arr.dtype=}")

'''
This function creates a dictionary for metadata used in training
The dictionary outputs an encoded label (e.g. [0, 0, 1]) when given a text label
'''
def create_polygon_map():
    lbs = ['REAL', 'FINE', 'COARSE']
    meta_map = {}
    for i, lb in enumerate(lbs):
        arr = np.zeros(len(lbs))
        arr[i] = 1
        meta_map[lb] = arr
    return meta_map

'''
Only binary version
'''
def create_polygon_map_retina():
    lbs = ['FINE', 'COARSE']
    meta_map = {}
    for i, lb in enumerate(lbs):
        arr = np.zeros(len(lbs))
        arr[i] = 1
        meta_map[lb] = arr
    return meta_map

'''
This function prepares image data, making in-memory dataset, 
since the dataset size allows it
Returns the normilized [0-1] numpy array of images, masks and list of metalabels (for train split)
'''
def prepare_data(split, root_path, img_folder = "img", mask_folder = "mask", labels = "metadata.csv", as_gray = False, resize_required = True, size = (256, 256), normalize = True):
    assert split == "train" or split == "val" or split == "test", "Only train/val/test splits are allowed"
    imgpath = os.path.join(root_path, img_folder)
    maskpath = os.path.join(root_path, mask_folder)
    imglist, masklist, lbs = [], [], []
    if split in ("val", "test"):
        files = sorted(os.listdir(imgpath)) 
        lbs = ["REAL"] * len(files) # validation and test data should be assosiated with one type of metadata
    else:
        metapath = os.path.join(root_path, labels)
        df = pd.read_csv(metapath)
        files, lbs = df["files"].to_list(), df["labels"].to_list()

    for fl in files:
        fpath = os.path.join(imgpath, fl)
        mpath = os.path.join(maskpath, fl)
        
        # images are loaded with 4 channesl 
        # io rescales images in range [0,1] when as_gray = True
        img = io.imread(fpath,  as_gray=as_gray) 
        mask = io.imread(mpath)  # masks are loaded in 2d format
        if resize_required:
            img = resize(img, (256, 256), anti_aliasing=True, preserve_range = True).astype(np.uint8)
            mask = resize(mask, (256, 256), anti_aliasing=True, preserve_range = True).astype(np.uint8)
        imglist.append(img)
        masklist.append(mask)
    
    img_array = np.stack(imglist)
    mask_array = np.stack(masklist)
    
    if as_gray: # images read as greyscale are normalized and squeezed in channel dimension
        # expanding channel dimmension as preferred by PyTorch
        img_array = np.expand_dims(img_array, -1)
    else:
        img_array = img_array[...,:3] # removing redundunt alpha channel
    
    if normalize:
        img_array = img_array/255 # type: ignore
        mask_array = (mask_array > 0).astype(np.uint8)
    return img_array, mask_array, lbs

def prepare_retina_data(split, root_path, data_distrib_path, as_gray = False, normalize = True):
    assert split == "train" or split == "val" or split == "test", "Only train/val/test splits are allowed"
    df = pd.read_csv(data_distrib_path)
    df = df[df["split"] == split]
    #imglist, masklist, lbs = df["images"].to_list(), df["masks_used"].to_list(), df["metalabels"].to_list()
    imglist, masklist = [], []
    for fl, mfl in zip(df["images"].to_list(), df["masks_used"].to_list()):
        fpath = os.path.join(root_path, fl)
        mpath = os.path.join(root_path, mfl)        
        
        # io rescales images in range [0,1] when as_gray = True
        img = io.imread(fpath,  as_gray=as_gray) 
        mask = io.imread(mpath)  # masks are loaded in 2d format
        imglist.append(img)
        masklist.append(mask)
    
    img_array = np.stack(imglist)
    mask_array = np.stack(masklist)    
    if normalize:
        img_array = img_array/255 # type: ignore
        mask_array = (mask_array > 0).astype(np.uint8)
    return img_array, mask_array, df["metalabels"].to_list()      
        
