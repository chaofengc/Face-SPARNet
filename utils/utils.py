import torch
import numpy as np
import cv2 as cv
from skimage import io
from PIL import Image
import os
import subprocess


def img_to_tensor(img_path, device, size=None, mode='rgb'):
    """
    Read image from img_path, and convert to (C, H, W) tensor in range [-1, 1]
    """
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    if mode=='bgr':
        img = img[..., ::-1]
    if size:
        img = cv.resize(img, size)
    img = img / 255 * 2 - 1 
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device) 
    return img_tensor.float()


def tensor_to_img(tensor, save_path=None, size=None, mode='rgb', normal=False):
    img_array = tensor.squeeze().data.cpu().numpy()
    img_array = img_array.transpose(1, 2, 0)
    if size is not None:
        img_array = cv.resize(img_array, size, interpolation=cv.INTER_LINEAR)
    if normal:
        #  img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        img_array = (img_array + 1.) / 2. * 255.
        img_array = img_array.clip(0, 255)
    if save_path:
        if img_array.max() <= 1:
            img_array = (img_array * 255).astype(np.uint8)
        io.imsave(save_path, img_array)

    return img_array.astype(np.uint8)


def tensor_to_numpy(tensor):
    return tensor.data.cpu().numpy()


def batch_numpy_to_image(array, size=None):
    """
    Input: numpy array (B, C, H, W) in [-1, 1]
    """
    if isinstance(size, int):
        size = (size, size)

    out_imgs = []
    array = np.clip((array + 1)/2 * 255, 0, 255) 
    array = np.transpose(array, (0, 2, 3, 1))
    for i in range(array.shape[0]):
        if size is not None:
            tmp_array = cv.resize(array[i], size)
        else:
            tmp_array = array[i]
        out_imgs.append(tmp_array)
    return np.array(out_imgs)


def batch_tensor_to_img(tensor, size=None):
    """
    Input: (B, C, H, W) 
    Return: RGB image, [0, 255]
    """
    arrays = tensor_to_numpy(tensor)
    out_imgs = batch_numpy_to_image(arrays, size)
    return out_imgs 


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def get_gpu_memory_map():
    """Get the current gpu usage within visible cuda devices.

    Returns
    -------
    Memory Map: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    Device Ids: gpu ids sorted in descending order according to the available memory.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ]).decode('utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible_devices = sorted([int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')])
    else: 
        visible_devices = range(len(gpu_memory))
    gpu_memory_map = dict(zip(range(len(visible_devices)), gpu_memory[visible_devices]))
    return gpu_memory_map, sorted(gpu_memory_map, key=gpu_memory_map.get)


