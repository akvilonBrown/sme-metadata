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
    
    transforms_training = Compose([
        MoveAxis()
    ])
    transforms_validation = transforms_training

    # random seed    
    set_seed(args.random_seed)    

    train_data, train_masks, train_idx = du.prepare_retina_data(split = "train", 
                                                                root_path = args.root_dir, 
                                                                data_distrib_path = args.datafile, 
                                                                as_gray = False, 
                                                                normalize = True)
    
    train_idx = np.array(train_idx)
    ixx = (train_idx == 'FINE')
    train_idx = list(train_idx[ixx])
    train_data = train_data[ixx]
    train_masks = train_masks[ixx]


    val_data, val_masks, val_idx = du.prepare_retina_data(split = "val", 
                                                                root_path = args.root_dir, 
                                                                data_distrib_path = args.datafile, 
                                                                as_gray = False, 
                                                                normalize = True)
                                                    



    du.print_info(train_data)
    du.print_info(train_masks)

    du.print_info(val_data)
    du.print_info(val_masks)
    
    
    # this is dictionary defining how to encode matalabels in dataset (ignored for reference models)
    metamap = du.create_polygon_map_retina() 
    print(f"{metamap = }")  
    
    dataset_train = CustomDataSet(inputs=train_data,
                                        targets=train_masks,
                                        metalabels=train_idx,
                                        mmap = metamap,
                                        transform=transforms_training)   

    
    
    dataset_val = CustomDataSet(inputs=val_data,
                                        targets=val_masks,
                                        metalabels=val_idx,
                                        mmap = metamap,
                                        transform=transforms_validation)
    
    print(f"{len(dataset_train) = }") 
    print(f"{len(dataset_val) = }")

    set_seed(args.random_seed) 
    # dataloader training
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=args.batch_size,
                                    shuffle=True)
    
    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_val,
                                       batch_size=args.batch_size,
                                       shuffle=False)
    
    print(f"{len(dataloader_training) = }")
    print(f"{len(dataloader_validation) = }")
    print("Data prepared")
    
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("cpu device engaged")
    
    model = models.UNet3SEb(n_channels=3, n_classes=2, bilinear=False).to(device) # type: ignore
    
    print("Model created")

    criterion = torch.nn.CrossEntropyLoss()
    
    # optimizer
    #lr = 0.0002
    #up_lr =0.0008
    #num_epochs = 100 
    steps = len(dataset_train)//args.batch_size
    print("Steps: " + str(steps))
    s_up = steps * args.num_epochs//args.cycles #15  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=args.up_lr, step_size_up=s_up,mode="triangular2", cycle_momentum=False)
    
    print("Optimizer and lr_scheduler set")
    
    # trainer
    trainer_object = Trainer_lr(model=model,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    training_DataLoader=dataloader_training,
                    validation_DataLoader=dataloader_validation,
                    lr_scheduler=lr_sheduler, # type: ignore
                    epochs=args.num_epochs,
                    epoch=0,
                    with_meta = False,
                    notebook=False,
                    best_model_dir = args.folder, #'saved_models'
                    saving_milestone = args.milestone, #10
                    save_name = args.save_name) #"best_model_refsub"

    
    # start training
    training_losses, validation_losses, lr_rates = trainer_object.run_trainer()
    print("Training completed") 
    
    data = {'train_losses': training_losses, 
            'val_losses': validation_losses}
    df = pd.DataFrame(data)
    df.to_csv (args.loss_log + '.csv', index = None, header=True) # type: ignore

    data = {'lr_rates': lr_rates}
    df = pd.DataFrame(data)
    df.to_csv (args.lr_log + '.csv', index = None, header=True) # type: ignore       
    
    fig = plot_training(training_losses, validation_losses, lr_rates, gaussian=True, sigma=1, figsize=(10, 4))
    plt.savefig(args.savefig + '.png')
    plt.savefig(args.savefig + '.pdf')  
    print("Plots saved") 

if __name__ == '__main__':
    args = cli() 
    main(args)
