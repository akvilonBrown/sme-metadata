import torch

import numpy as np
from tqdm import tqdm


import torch
import numpy as np
from tqdm import tqdm


def predict_batch(
    model,
    test_loader,
    device,
    notebook=False,
    dtype = "uint8",
    with_meta = False,
    regression = False
) -> np.ndarray:
    """
    Takes in images and outputs final preditions
        :param model: a PyTorch Model object
        :param test_loader: torch dataloader
        :param notebook: bool to define the representation of progress bar 
        :param with_meta: bool - to predict with metadata or not  
        :param regression: bool - instructs to output raw probability maps     
        :return: predicted full-sized images
        
    """
 
    #num_channels = img_data.shape[-1] 

    try:
        tqdm._instances.clear()  # type: ignore
    except:
        pass

    if notebook:
        from tqdm.notebook import tqdm, trange
    else:
        from tqdm import tqdm, trange
    
    final_full_preds = []
    
    model.eval()
    batch_iter = tqdm(enumerate(test_loader), 'Prediction', total=len(test_loader),
                     leave=False)
    
    for i, (x, m, y) in batch_iter:        
        
        x  = x.to(device)
        
        with torch.no_grad():
            if with_meta:
                m = m.to(device)
                out_test = model(x, m)
            else:
                out_test = model(x)
                
        if regression:
            out_test = out_test.cpu().numpy()
        else:    
            out_test = torch.softmax(out_test, dim=1)
            out_test = torch.argmax(out_test, dim=1).cpu().numpy()        
        final_full_preds.append(out_test)
    final_full_preds = np.concatenate(final_full_preds)

    final_full_preds = final_full_preds.astype(dtype = dtype)
    return final_full_preds
       

