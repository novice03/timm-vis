from timm_vis.helpers import *
from timm_vis.methods import *

class Visualizer():
    def __init__(model):
        self.model = model
    
    def visualize_filters(filter_name = None, max_filters = 64, size = 128, figsize = (16, 16), save_path = None):
        return visualize_filters(model = self.model, filter_name = filter_name, max_filters = max_filters, 
                                    size = size, figsize = figsize, save_path = save_path)
        