# Imports

import numpy as np
import pandas as pd
import os

import utils.data_utils as du
from utils.cli import cli
from utils.transformations import Compose, MoveAxis
from utils.dataset import CustomDataSet
from torch.utils.data import DataLoader
import models
import torch
from utils.trainer import Trainer_lr
#import time
import random
from utils.visual import plot_training
from utils.prediction import predict_batch
from utils.metrics import *
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False # type: ignore
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):

    for arg in vars(args):
        print(arg, getattr(args, arg)) 

    transforms_test = Compose([
        MoveAxis()
    ])    

    # random seed
    set_seed(args.random_seed)     

    test_data, test_masks, test_idx = du.prepare_retina_data(split = "test", 
                                                                root_path = args.root_dir, 
                                                                data_distrib_path = args.datafile, 
                                                                as_gray = False, 
                                                                normalize = True)                                                  
                                   



    du.print_info(test_data)
    du.print_info(test_masks)
    
    
    # this is dictionary defining how to encode matalabels in dataset (ignored for reference models)
    metamap = du.create_polygon_map_retina() 
    print(f"{metamap = }")  
    
    dataset_test = CustomDataSet(inputs=test_data,
                                        targets=test_masks,
                                        metalabels=test_idx,
                                        mmap = metamap,
                                        transform=transforms_test)

    
    print(f"{len(dataset_test) = }")  

    set_seed(args.random_seed) 
    # dataloader test
    dataloader_test = DataLoader(dataset=dataset_test,
                                    batch_size=args.batch_size,
                                    shuffle=False)
    

    
    print(f"{len(dataloader_test) = }")
    print("Data prepared")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("cpu device engaged")
    
    model_restored = models.UNet3SEb(n_channels=3, n_classes=2, bilinear=False).to(device) # type: ignore         
    model_name = args.folder + '/' + args.save_name +  '.pt'
    model_weights = torch.load(model_name)    
    model_restored.load_state_dict(model_weights)
    print("Model restored")
    
    predictions = predict_batch(model_restored, dataloader_test, device, with_meta = False, regression = False, dtype = "uint8" )
    predictions_renorm = predictions*255
    du.print_info(predictions)
    du.print_info(predictions_renorm)

    iou = iou_score_total(test_masks, predictions)
    dice = dice_score_total(test_masks, predictions)
    iou_list = iou_score_piecewise(test_masks, predictions)
    dice_list = dice_score_piecewise(test_masks, predictions)
    ixx =  ["total"] +  list(range(len(iou_list)))

    print(f"{iou = }, {dice = }")    
    data = {
            "iou": [iou]+iou_list,
            "dice": [dice] + dice_list   
            }

    df = pd.DataFrame(data, index = ixx)
    df.to_csv (args.savemetrics + '.csv', header=True)

if __name__ == '__main__':
    args = cli() 
    main(args)

