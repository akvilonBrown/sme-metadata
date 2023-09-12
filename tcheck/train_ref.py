# Imports

import numpy as np
import pandas as pd
import os

import utils.data_utils as du
from utils.transformations import Compose, MoveAxis
from utils.dataset import CustomDataSet
from torch.utils.data import DataLoader
import models
import torch
from utils.trainer import Trainer_lr
#import inference
#import time
import random
#from inference import plot_training
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False # type: ignore
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():

    folder = 'saved_models'
    #path = os.path.join(start_path, folder)
    batch_size=8
    root_dir_train = "/filer-5/user/plutenko/medical/data/train"
    root_dir_val = "/filer-5/user/plutenko/medical/data/val"
    imdir = "img"
    maskdir = "mask"
    label_file = "metadata.csv"   

    transforms_training = Compose([
        MoveAxis()
    ])
    transforms_validation = transforms_training

    # random seed
    random_seed = 42
    set_seed(random_seed)
    
    train_data, train_masks, train_idx = du.prepare_data(split = "train", root_path = root_dir_train, img_folder = imdir, mask_folder = maskdir, labels = label_file)
    val_data, val_masks, val_idx = du.prepare_data(split = "val", root_path = root_dir_val, img_folder = imdir, mask_folder = maskdir, labels = label_file)

    du.print_info(train_data)
    du.print_info(train_masks)

    du.print_info(val_data)
    du.print_info(val_masks)
    
    
    # this is dictionary defining how to encode matalabels in dataset (ignored for reference models)
    metamap = du.create_polygon_map() 
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

    set_seed(random_seed)
    # dataloader training
    dataloader_training = DataLoader(dataset=dataset_train,
                                    batch_size=batch_size,
                                    shuffle=True)
    
    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_val,
                                       batch_size=batch_size,
                                       shuffle=False)
    
    print(f"{len(dataloader_training) = }")
    print(f"{len(dataloader_validation) = }")
    print("Data prepared")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("cpu device engaged")
    
    model = models.UNet3SEb(n_channels=1, n_classes=2, bilinear=False).to(device) # type: ignore
    
    print("Model created")

    criterion = torch.nn.CrossEntropyLoss()
    
    # optimizer
    lr = 0.0002
    up_lr =0.0008
    num_epochs = 100 
    steps = len(dataset_train)//batch_size
    print("Steps: " + str(steps))
    s_up = steps * num_epochs//15  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_sheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=up_lr, step_size_up=s_up,mode="triangular2", cycle_momentum=False)
    
    print("Optimizer and lr_scheduler set")
    
    # trainer
    trainer_object = Trainer_lr(model=model,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    training_DataLoader=dataloader_training,
                    validation_DataLoader=dataloader_validation,
                    lr_scheduler=lr_sheduler, # type: ignore
                    epochs=num_epochs,
                    epoch=0,
                    with_meta = False,
                    notebook=False,
                    best_model_dir = folder, #'saved_models'
                    saving_milestone = 10,
                    save_name = "best_model_ref")

    
    # start training
    training_losses, validation_losses, lr_rates = trainer_object.run_trainer()


    data = {'train_losses': training_losses, 
            'val_losses': validation_losses}
    df = pd.DataFrame(data)
    df.to_csv ('losses_ref.csv', index = None, header=True) # type: ignore

    data = {'lr_rates': lr_rates}
    df = pd.DataFrame(data)
    df.to_csv ('lr_rates_ref.csv', index = None, header=True) # type: ignore
    print("Training completed")
    
    '''
    fig = plot_training(training_losses, validation_losses, validation_losses, gaussian=True, sigma=1, figsize=(10, 4))
    plt.savefig('rate.png')
    plt.savefig('rate.pdf')
    plt.plot(lr_rates)
    plt.savefig('rate2.png')
    plt.savefig('rate2.pdf')   
    print("Plots saved") 

    model_restored = models.UNet3SEbMetaSE(n_channels=1, n_classes=2, meta_length = 3, bilinear=False).to(device)
    model_name = 'saved_models/best_model.pt'
    model_weights = torch.load(model_name)    
    model_restored.load_state_dict(model_weights)

    test_data, test_masks, test_list = dataloaders.get_poly_data('test')    
    metamap = dataloaders.create_polygon_map() 
    res = inference.predict_metamodel_fs(model_restored, test_data, test_list, metamap, device, mode = "REAL")    
    inference.calc_metrics(test_masks, res) 
    '''
if __name__ == '__main__':
    main()
