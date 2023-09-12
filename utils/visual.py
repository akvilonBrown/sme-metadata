import numpy as np
import matplotlib.pyplot as plt

def plot_training(training_losses,
                  validation_losses,
                  learning_rate,
                  gaussian=True,
                  sigma=2,
                  figsize=(8, 6)
                  ):
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """


    from matplotlib import gridspec
    from scipy.ndimage import gaussian_filter

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

    subfig1 = fig.add_subplot(grid[0, 0])
    subfig2 = fig.add_subplot(grid[0, 1])

    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)

        linestyle_original = '.'
        color_original_train = 'lightcoral'
        color_original_valid = 'lightgreen'
        color_smooth_train = 'red'
        color_smooth_valid = 'green'
        alpha = 0.25
    else:
        linestyle_original = '-'
        color_original_train = 'red'
        color_original_valid = 'green'
        alpha = 1.0

    # Subfig 1
    subfig1.plot(x_range, training_losses, linestyle_original, color=color_original_train, label='Training',
                 alpha=alpha)
    subfig1.plot(x_range, validation_losses, linestyle_original, color=color_original_valid, label='Validation',
                 alpha=alpha)
    if gaussian:
        subfig1.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig1.plot(x_range, validation_losses_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig1.title.set_text('Training & validation loss')
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('Loss')

    subfig1.legend(loc='upper right')

    # Subfig 2
    x_range2 = list(range(1, len(learning_rate) + 1)) 
    subfig2.plot(x_range2, learning_rate, color='black')
    subfig2.title.set_text('Learning rate')
    subfig2.set_xlabel('Cycle')
    subfig2.set_ylabel('LR')

    return fig

def show_image_pair(im1, im2, cmap = 'gray', figsize=(12,8)):    
     
    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(im1, cmap = cmap)
    plt.title('image1')
    
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(im2, cmap = cmap)
    plt.title('image2')

    plt.tight_layout()
    plt.show()

def show_examples(name: str, image: np.ndarray, mask: np.ndarray, 
                         gt: np.ndarray, cline: str, figsize=(10, 14), cmap = 'gray', 
                         imtitle = "Image", msk1_title = "Mask", msk2_title = "gt"):
    plt.figure(figsize = figsize)
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap = cmap)
    plt.axis('off')
    plt.title(f"{imtitle}: {name} meta: {cline}")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap = cmap)
    plt.axis('off')
    plt.title(f"{msk1_title}")
    
    plt.subplot(1, 3, 3)
    plt.imshow(gt, cmap = cmap)
    plt.axis('off')
    plt.title(f"{msk2_title}")
    
    plt.tight_layout()
    plt.show()

def overlap(gtim, img):     
    nim = np.zeros(gtim.shape + (3,), dtype = np.uint8 )
    cross = (gtim&img).astype(np.uint8)*255
    nim[..., 0] = cross 
    nim[..., 1] = cross  
    nim[..., 2] = cross
    nim[..., 0] = nim[..., 0] + np.where(img>gtim, 255, 0)
    nim[..., 2] = nim[..., 2] + np.where(gtim>img, 255, 0)  
    return nim