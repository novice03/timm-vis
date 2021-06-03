## PyTorch Image Models Visualizer

Implementation of various visualization techniques for pytorch image classifiers. This library can be used to visualize and understand any PyTorch image classifier. This is NOT an official PyTorch library, nor is it affiliated with Ross Wightman's [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) library. [details.ipynb](https://github.com/novice03/timm-vis/blob/main/details.ipynb) has visual examples of all methods implemented.

Currently, the following methods are implemented:

- Filter visualization
- Activations visualization
- Maximally activated patches 
- Saliency maps [1]
- Synthetic image generation [1]
- Adversarial attacks to fool models 
- Feature inversion [2]
- Grad-CAM [3]
- Deep Dream [4]

Specific examples and details about the implementation and parameters of the above methods are described in details.ipynb. All of the above visualization techniques are discussed in [this](https://www.youtube.com/watch?v=6wcs6szJWMY&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=14)
lecture.

## Installation

```bash
$ pip install timm-vis
```

## Usage

```python
from timm_vis.methods import *

# available methods - visualize_filters, visualize_activations, 
#   maximally_activated_patches, saliency_map, 
#   generate_image, fool_model, feature_inversion, deep_dream

```

## Paper References

[1] Karen Simonyan, Andrea Vedaldi, Andrew Zisserman. Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. [https://arxiv.org/abs/1312.6034](https://arxiv.org/abs/1312.6034). 

[2] Aravindh Mahendran, Andrea Vedaldi. Understanding Deep Image Representations by Inverting Them [https://arxiv.org/abs/1412.0035](https://arxiv.org/abs/1412.0035)

[3] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization [https://arxiv.org/abs/1610.02391 (https://arxiv.org/abs/1610.02391)]

[4] Alexander Mordvintsev, Christopher Olah, Mike Tyka. Inceptionism: Going Deeper into Neural Networks [https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)

## Code References

[5] Ross Wightman [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

[6] Irfan Alghani Khalid [Saliency Map for Visualizing Deep Learning Model Using PyTorch](https://towardsdatascience.com/saliency-map-using-pytorch-68270fe45e80)

[7] Utku Ozbulak. [pytorch-cnn-adversarial-attacks](https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks)

[8] Duc Ngo [deep-dream-in-pytorch](https://github.com/duc0/deep-dream-in-pytorch)
