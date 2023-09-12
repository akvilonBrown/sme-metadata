import numpy as np
from skimage.transform import resize
#from sklearn.externals._pilutil import bytescale
#import albumentations as A

def normalize_01(inp: np.ndarray):
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    return inp_out

def normalize_01_simple(inp: np.ndarray):
    inp_out = inp / 255
    return inp_out

def normalize(inp: np.ndarray, mean: float, std: float):
    inp_out = (inp - mean) / std
    return inp_out

'''
def re_normalize(inp: np.ndarray,
                 low: int = 0,
                 high: int = 255
                 ):
    """Normalize the data to a certain range. Default: [0-255]"""
    inp_out = bytescale(inp, low=low, high=high)
    return inp_out
'''

class Compose:
    """
    Composes several transforms together.
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, inp, target):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target

    def __repr__(self): return str([transform for transform in self.transforms])


class MoveAxis:
    """From [H, W, C] to [C, H, W]"""

    def __init__(self, transform_input: bool = True, transform_target: bool = False):
        self.transform_input = transform_input
        self.transform_target = transform_target

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        if self.transform_input: inp = np.moveaxis(inp, -1, 0)
        if self.transform_target: tar = np.moveaxis(tar, -1, 0)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Normalize01:
    """Squash image input to the value range [0, 1] (no clipping)"""

    def __init__(self):
        pass

    def __call__(self, inp, tar):
        inp = normalize_01(inp)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class Normalize01_simple:
    """Squash image input to the value range [0, 1] (no clipping)"""

    def __init__(self):
        pass

    def __call__(self, inp, tar):
        inp = normalize_01_simple(inp)
        tar = normalize_01_simple(tar)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Normalize:
    """Normalize based on mean and standard deviation."""

    def __init__(self,
                 mean: float,
                 std: float
                 ):
        self.mean = mean
        self.std = std

    def __call__(self, inp, tar):
        inp = normalize(inp, self.mean, self.std)

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

'''
# These classes below require albumentations lib

class HorizontalFlip:
    """Wrapper around Albumentation HorizontalFlip"""

    def __init__(self, width=512, height=512, transform_input: bool = True, transform_target: bool = True, p = 0.5):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.flipper = A.HorizontalFlip(p = p)

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        flipped_dict = self.flipper(image=inp, mask=tar)
        if self.transform_input: inp = flipped_dict['image']
        if self.transform_target: tar = flipped_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})

class RandomScale:
    """Wrapper around Albumentation HorizontalFlip"""

    def __init__(self, width=512, height=512, transform_input: bool = True, 
                 transform_target: bool = True, 
                 scale_limit = [0.7, 1.],
                 p = 0.5):
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.former = A.RandomScale(scale_limit = scale_limit, 
                                       interpolation=0, #cv2.INTER_NEAREST
                                       p = p)
       

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        form_dict = self.former(image=inp, mask=tar)
        if self.transform_input: inp = form_dict['image']
        if self.transform_target: tar = form_dict['mask']

        return inp, tar

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})
'''        