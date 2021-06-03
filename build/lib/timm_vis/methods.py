import math
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from timm_vis.helpers import *

def visualize_filters(model, filter_name = None, max_filters = 64, size = 128, figsize = (16, 16), save_path = None):
    """
        Plots filters of a convolutional layer by interpreting them as grayscale images
    """

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

def visualize_activations(model, module, img_path, max_acts = 64, rgb = True, figsize = (16, 16), save_path = None):
    """
        Plots the activations of a module recorded during a forward pass on an image
    """

    img_t = preprocess_image(img_path, rgb = rgb)
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

def maximally_activated_patches(model, img_path, patch_size = 448, stride = 100, num_patches = 5, rgb = True, 
                                    figsize = (16, 16), device = 'cuda', save_path = None):

    """
        Plots the patches of an image that produce the highest activations
    """

    model.eval()
    model.to(device)
    
    img = Image.open(img_path)
    img_mean = int(np.mean(np.asarray(img)))
    img_t = preprocess_image(img_path, rgb = rgb).to(device)
    
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
        if rgb:
            draw.rectangle([x0, y0, x1, y1], fill = (img_mean, img_mean, img_mean))
        else:
            draw.rectangle([x0, y0, x1, y1], fill = (img_mean))
        
        occ_img_t = preprocess_image(img_copy, rgb = rgb).to(device)
        
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
        
        if rgb:
            axs[plot_idx].imshow(np.asarray(img_copy.crop((x0, y0, x1, y1))))
        else:
            axs[plot_idx].imshow(np.asarray(img_copy.crop((x0, y0, x1, y1))), cmap = 'gray')
        axs[plot_idx].set_yticks([])
        axs[plot_idx].set_xticks([])

def saliency_map(model, img_path, rgb = True, figsize = (16, 16), device = 'cuda', save_path = None):
    """
        Plots the gradient of the score of the predicted class with respect to image pixels
    """

    model.eval()
    model.to(device)
    
    img = Image.open(img_path)
    img_t = preprocess_image(img, rgb = rgb).to(device)
    img_np = scale(img_t.detach().cpu()[0].permute(1, 2, 0).numpy())
    img_t.requires_grad = True
    
    out = model(img_t)
    max_out = out.max()
    max_out.backward()    
    saliency, _ = torch.max(img_t.grad.data.abs(), dim = 1)
    saliency = saliency.squeeze(0)
    saliency_img = saliency.detach().cpu().numpy()
    
    fig, axs = plt.subplots(1, 2, figsize = figsize)
    
    if rgb:
        axs[0].imshow(img_np)
    else:
        axs[0].imshow(img_np, cmap = 'gray')
        
    axs[1].imshow(saliency_img, cmap = 'gray')
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    
    if save_path:
        plt.savefig(save_path)

def generate_image(model, target_class, epochs, min_prob, lr, weight_decay, step_size = 100, gamma = 0.6,
                        noise_size = 224, p_freq = 50, init = torch.randn, device = 'cuda', figsize = (6, 6), save_path = None):
    
    """
        Starting from a random initialization, generates an image that maximizes the score for a specific class using
        gradient ascent
    """

    name, weights = next(model.named_parameters())
    in_size = weights.size()[1]
    
    noise = init([1, in_size, noise_size, noise_size]).to(device)
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
    
    rgb = in_size > 1
    fig, axs = plt.subplots(1, figsize = figsize)
    img_np = postprocess_image(noise, rgb = rgb)
    if rgb:
        axs.imshow(img_np)
    else:
        axs.imshow(img_np, cmap = 'gray')
    axs.set_xticks([])
    axs.set_yticks([])

    if save_path:
        fig.savefig(save_path)
    
    return noise

def fool_model(model, img_path, target_class, epochs, min_prob, lr, step_size, gamma,   
                        p_freq = 50, device = 'cuda', rgb = True, figsize = (6, 6), save_path = None):
    
    """
        Modifies a given image to have a high score for a specific class, similar to generate_image()
    """

    orig_img = preprocess_image(img_path, rgb = rgb).to(device)
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
    img_np = postprocess_image(orig_img, rgb = rgb)
    
    if rgb:    
        axs.imshow(img_np)
    else:
        axs.imshow(img_np, cmap = 'gray')
    axs.set_xticks([])
    axs.set_yticks([])

    if save_path:
        fig.savefig(save_path)
    
    return orig_img

def feature_inversion(model, modules, img_path, epochs, lr, step_size = 100, gamma = 0.6, mu = 1e-1, 
                          device = 'cuda', rgb = True, figsize = (16, 16), save_path = None):
    
    """
        Reconstructs an image based on its feature representation at various modules
    """

    orig_img = preprocess_image(img_path, rgb = rgb).to(device)
    model = model.to(device)
    model.eval()
    recreated_imgs = []
    
    for module in modules:   
        recreated_imgs.append(feature_inversion_helper(model, module, orig_img, epochs = epochs,
                                lr = lr, step_size = step_size, gamma = gamma, mu = mu, device = device))
        
    fig, axs = plt.subplots(1, len(recreated_imgs), figsize = figsize)
    
    for i in range(len(recreated_imgs)):
        if rgb:
            axs[i].imshow(postprocess_image(recreated_imgs[i], rgb = rgb))
        else:
            axs[i].imshow(postprocess_image(recreated_imgs[i], rgb = rgb), cmap = 'gray')

    if save_path:
        fig.savefig(save_path)

def feature_inversion_helper(model, module, orig_img, epochs, lr, step_size, gamma, mu, noise_size = 224, init = torch.randn, 
                                device = 'cuda'):
    
    """
        Performs feature inversion on one module
    """

    acts = [0]    
    def hook_fn(self, input, output):
        acts[0] = output
        
    handle = module.register_forward_hook(hook_fn)
    _ = model(orig_img)
    orig_features = acts[0]
    name, weights = next(model.named_parameters())
    in_size = weights.size()[1]
    
    noise = init([1, in_size, noise_size, noise_size]).to(device)
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

def grad_cam(model, module, img_path, class_id, rgb = True, device = 'cuda', alpha = 0.6, 
                 figsize = (16, 16), save_path = None):
    
    acts = [0]
    grads = [0]
    
    def f_hook(self, input, output):
        acts[0] = output

    def b_hook(self, grad_in, grad_out):
        grads[0] = grad_out
    
    h1 = module.register_forward_hook(f_hook)
    h2 = module.register_backward_hook(b_hook)
    
    img = Image.open(img_path)
    img_t = preprocess_image(img_path, rgb = rgb).to(device)
    model.to(device)
    outs = model(img_t)
    h1.remove()
    h2.remove()
    outs[0, class_id].backward()
    
    gap = torch.mean(grads[0][0].view(grads[0][0].size(0), grads[0][0].size(1), -1), dim = 2)
    acts = acts[0][0]
    gradcam = torch.nn.ReLU()(torch.sum(gap[0].reshape((gap.size()[1], 1, 1)) * acts, dim = 0))
    arr = transforms.Resize((img.size[1], img.size[0]))(gradcam.unsqueeze(0))
    gradcam_img = arr.detach().cpu().permute((1, 2, 0)).squeeze(-1)

    fig, axs = plt.subplots(1, figsize = figsize)
    
    axs.imshow(np.asarray(img))
    axs.imshow(gradcam_img, alpha = alpha)
    axs.set_xticks([])
    axs.set_yticks([])
    
    if save_path:
        fig.savefig(save_path)

def deep_dream(model, module, img_path, epochs, lr, step_size = 100, gamma = 0.6, device = 'cuda', rgb = True,
                   figsize = (12, 12), save_path = None):

    """
        Modifies the input image to maximize activation at a specific module
    """
    
    img_t = preprocess_image(img_path, rgb = rgb).to(device)
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
    img_np = postprocess_image(img_t, rgb = rgb)
    
    if rgb:
        axs.imshow(img_np)
    else:
        axs.imshow(img_np, cmap = 'gray')
        
    axs.set_xticks([])
    axs.set_yticks([])

    if save_path:
        fig.savefig(save_path)
    
    return img_t