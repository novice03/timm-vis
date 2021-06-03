import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageDraw

pre_transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),                 
                                 transforms.Resize((224, 224))])

post_transforms = transforms.Compose([transforms.Normalize(mean = [0, 0, 0], std = [1/0.229, 1/0.224, 1/0.225]),                 
                              transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [1, 1, 1])])

pre_transforms_gray = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean = [0.449], std = [0.226]),                 
                                 transforms.Resize((224, 224))])

post_transforms_gray = transforms.Compose([transforms.Normalize(mean = [0], std = [1/0.226]),                 
                              transforms.Normalize(mean = [-0.449], std = [1])])

def scale(arr):
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')

def preprocess_image(img_path, rgb = True):
    if isinstance(img_path, str):
        img = Image.open(img_path)
    else:
        img = img_path
    
    if rgb:
        img_t = pre_transforms(img).unsqueeze(0)
    else:
        img_t = pre_transforms_gray(img).unsqueeze(0)
    return img_t

def postprocess_image(img_t, rgb = True):
    if rgb:
        img_t = post_transforms(img_t[0])
    else:
        img_t = post_transforms_gray(img_t[0])        
        
    img_np = img_t.detach().cpu().numpy().transpose(1,2,0)
    img_np = scale(np.clip(img_np, 0, 1))
    
    return img_np

def gen_coords(i, patch_size, stride, dim1, dim2):
    x0 = int(stride * (i % dim1))
    y0 = int(stride * int(i / dim2))
    x1 = x0 + patch_size
    y1 = y0 + patch_size
    
    return x0, y0, x1, y1

def total_variation_regularizer(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))