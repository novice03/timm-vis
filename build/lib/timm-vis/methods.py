import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
from torch.nn.functional import softmax
from torchvision import transforms
import timm

def visualize_filters(model, filter_name = None, max_filters = 64, size = 128, figsize = (16, 16), save_path = None):
    name, weights = next(model.named_parameters())
    
    for layer_name, layer_weights in model.named_parameters():
        if layer_name == filter_name:
            name = layer_name
            weights = layer_weights 
          
    w_size = weights.size()
    merged_weights = weights.reshape(w_size[0] * w_size[1], w_size[2], w_size[2]).detach().numpy()
    out_chs = merged_weights.shape[0]
    
    if out_chs > max_filters:
        merged_weights = merged_weights[torch.randperm(out_chs)[:max_filters]]
        out_chs = max_filters    
    
    sqrt = int(math.sqrt(out_chs))
    fig, axs = plt.subplots(sqrt, sqrt, figsize = figsize)
    
    if not size:
        size = merged_weights.shape[2]
    
    for i in range(sqrt ** 2):
        weight = merged_weights[i]
        scaled = scale(weight)
        resized = transforms.Resize((size, size))(Image.fromarray(scaled))
        plot_idx = int(i / sqrt), i % sqrt
        
        axs[plot_idx].imshow(resized, cmap = 'gray')
        axs[plot_idx].set_yticks([])
        axs[plot_idx].set_xticks([])
    
    if save_path:
        fig.savefig(save_path)

def visualize_activations(model, module, img_path, max_acts = 64, figsize = (16, 16), save_path = None):
    img_t = preprocess_image(img_path)
    acts = [0]

    def hook_fn(self, input, output):
        acts[0] = output
    
    handle = module.register_forward_hook(hook_fn)
    out = model(img_t)
    handle.remove()
    acts = acts[0][0].detach().numpy()
    
    if acts.shape[0] > max_acts:
        acts = acts[torch.randperm(acts.shape[0])[:max_acts]]
    
    sqrt = int(math.sqrt(acts.shape[0]))
    fig, axs = plt.subplots(sqrt, sqrt, figsize = figsize)
    
    for i in range(sqrt ** 2):
        scaled = scale(acts[i])
        
        plt_idx = int(i / sqrt), i % sqrt
        axs[plt_idx].imshow(scaled, cmap = 'gray')
        axs[plt_idx].set_yticks([])
        axs[plt_idx].set_xticks([])
    
    if save_path:
        fig.savefig(save_path)

def maximally_activated_patches(model, img_path, patch_size = 448, stride = 100, num_patches = 5, figsize = (16, 16),
                                    device = 'cuda', save_path = None):
    model.eval()
    model.to(device)
    
    img = Image.open(img_path).convert('RGB')
    img_mean = int(np.mean(np.asarray(img)))
    img_t = preprocess_image(img_path).to(device)
    
    with torch.no_grad():
        out = model(img_t)
        
    probs = softmax(out[0], dim = 0)
    max_index = probs.argmax()
    orig = probs[max_index]
    
    dim1 = int(((img.size[0] - patch_size) / stride) + 1)
    dim2 = int(((img.size[1] - patch_size) / stride) + 1)
    diff = []
    
    for i in range(dim1 * dim2):
        img_copy = img.copy()
        
        x0, y0, x1, y1 = gen_coords(i, patch_size, stride, dim1, dim2)
        
        draw = ImageDraw.Draw(img_copy)
        draw.rectangle([x0, y0, x1, y1], fill = (img_mean, img_mean, img_mean))
        
        occ_img_t = preprocess_image(img_copy).to(device)
        
        with torch.no_grad():
            out = model(occ_img_t)
            
        probs_occ = softmax(out[0], dim = 0)
        diff.append(abs(orig - probs_occ[max_index].item()))
    
    diff = np.array(diff)
    top_indices = diff.argsort()[-num_patches:]
    fig, axs = plt.subplots(int(num_patches / 5), min(num_patches, 5), figsize = figsize)
    
    for i, idx in enumerate(top_indices):
        img_copy = img.copy()
        x0, y0, x1, y1 = gen_coords(idx, patch_size, stride, dim1, dim2)
        
        if num_patches > 5:
            plot_idx = int(i / 5), i % 5
        else:
            plot_idx = i
        
        axs[plot_idx].imshow(np.asarray(img_copy.crop((x0, y0, x1, y1))))
        axs[plot_idx].set_yticks([])
        axs[plot_idx].set_xticks([])

def saliency_map(model, img_path, figsize = (16, 16), device = 'cuda', save_path = None):
    model.eval()
    model.to(device)
    
    img = Image.open(img_path).convert('RGB')
    img_t = preprocess_image(img).to(device)
    img_np = scale(img_t.detach().cpu()[0].permute(1, 2, 0).numpy())
    img_t.requires_grad = True
    
    out = model(img_t)
    max_out = out.max()
    max_out.backward()    
    saliency, _ = torch.max(img_t.grad.data.abs(), dim = 1)
    saliency = saliency.squeeze(0)
    saliency_img = saliency.detach().cpu().numpy()
    
    fig, axs = plt.subplots(1, 2, figsize = figsize)
    axs[0].imshow(img_np)
    axs[1].imshow(saliency_img, cmap = 'gray')
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    
    if save_path:
        plt.savefig(save_path)

def generate_image(model, target_class, epochs, min_prob, lr, weight_decay, step_size = 100, gamma = 0.6,
                        noise_size = 224, p_freq = 50, init = torch.randn, device = 'cuda', figsize = (6, 6)):
    
    
    noise = init([1, 3, noise_size, noise_size]).to(device)
    noise.requires_grad = True
    model = model.to(device)
    opt = torch.optim.SGD([noise], lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size, gamma = gamma)
    
    for i in range(1, epochs + 1):
        opt.zero_grad()
        outs = model(noise)
        p = softmax(outs[0], dim = 0)[target_class]
        
        if i % p_freq == 0 or i == epochs:        
            print('Epoch: {} Confidence score for class {}: {}'.format(i, target_class, p))
            
        if p > min_prob:
            print('Reached {} confidence score in epoch {}. Stopping early.'.format(p, i))
            break
            
        obj = - outs[0][target_class]
        obj.backward()
        opt.step()
        scheduler.step()
    
    fig, axs = plt.subplots(1, figsize = figsize)
    img_np = postprocess_image(noise)
    axs.imshow(img_np)
    axs.set_xticks([])
    axs.set_yticks([])
    
    return noise

def fool_model(model, img_path, target_class, epochs, min_prob, lr,
                        step_size, gamma, p_freq = 50, init = torch.randn, device = 'cuda', figsize = (6, 6)):
    
    orig_img = preprocess_image(img_path).to(device)
    orig_img.requires_grad = True
    model = model.to(device)
    opt = torch.optim.SGD([orig_img], lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size, gamma = gamma)
    
    for i in range(1, epochs + 1):
        opt.zero_grad()
        outs = model(orig_img)
        p = softmax(outs[0], dim = 0)[target_class]
        
        if i % p_freq == 0 or i == epochs:        
            print('Epoch: {} Confidence score for class {}: {}'.format(i, target_class, p))
            
        if p > min_prob:
            print('Reached {} confidence score in epoch {}. Stopping early.'.format(p, i))
            break
            
        obj = - outs[0][target_class]
        obj.backward()
        opt.step()
        scheduler.step()
    
    fig, axs = plt.subplots(1, figsize = figsize)
    img_np = postprocess_image(orig_img)
    axs.imshow(img_np)
    axs.set_xticks([])
    axs.set_yticks([])
    
    return orig_img

def feature_inversion_helper(module, orig_img, epochs, lr, step_size, gamma, mu, noise_size = 224, init = torch.randn, 
                                device = 'cuda'):
    acts = [0]    
    def hook_fn(self, input, output):
        acts[0] = output
        
    handle = module.register_forward_hook(hook_fn)
    _ = model(orig_img)
    orig_features = acts[0]
    
    noise = init([1, 3, noise_size, noise_size]).to(device)
    noise.requires_grad = True
    opt = torch.optim.SGD([noise], lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size, gamma = gamma)
    
    for i in range(epochs):
        opt.zero_grad()
        _ = model(noise)
        curr = acts[0]
        
        loss = ((orig_features - curr) ** 2).sum() + mu * total_variation_regularizer(noise)
        loss.backward(retain_graph = True)
        
        opt.step()
        scheduler.step()
    
    handle.remove()
    return noise

def feature_inversion(model, module_dict, img_path, epochs, lr, step_size = 100, gamma = 0.6, mu = 1e-1, 
                          device = 'cuda', figsize = (16, 16)):
    
    orig_img = preprocess_image(img_path).to(device)
    model = model.to(device)
    model.eval()
    recreated_imgs = []
    
    for module in modules:   
        recreated_imgs.append(feature_inversion_helper(module, orig_img, epochs = epochs,
                                lr = lr, step_size = step_size, gamma = gamma, mu = mu, device = device))
        
    fig, axs = plt.subplots(1, len(recreated_imgs), figsize = figsize)
    
    for i in range(len(recreated_imgs)):
        axs[i].imshow(postprocess_image(recreated_imgs[i]))

def deep_dream(model, module, img_path, epochs, lr, step_size = 100, gamma = 0.6, device = 'cuda', figsize = (12, 12)):
    
    img_t = preprocess_image(img_path).to(device)
    img_t.requires_grad = True
    opt = torch.optim.SGD([img_t], lr = lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size, gamma = gamma)
    acts = [0]

    def hook_fn(self, input, output):
        acts[0] = output

    model.to(device)
    model.eval()
    handle = module.register_forward_hook(hook_fn)  
    
    for i in range(epochs):
        opt.zero_grad()
        _ = model(img_t)
        loss = -acts[0].norm()
        loss.backward()
        opt.step()
        scheduler.step()
    
    handle.remove()
    
    fig, axs = plt.subplots(1, figsize = figsize)
    img_np = postprocess_image(img_t)
    axs.imshow(img_np)
    axs.set_xticks([])
    axs.set_yticks([])
    
    return img_t