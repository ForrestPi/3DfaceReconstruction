import numpy as np
import torch

def predict_vertices(params, roi_box, dense, transform=True):
    vertices = reconstruct_vertices(params, dense=dense)
