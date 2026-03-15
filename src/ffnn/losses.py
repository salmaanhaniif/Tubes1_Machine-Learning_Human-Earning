import numpy as np

try:
    from .autodiff import Node
except ImportError:
    from autodiff import Node

def mse(y_true, y_pred):
    pass

def categoricalCrossentropy(y_true, y_pred, from_softmax=False):
    pass

def binaryCrossentropy(y_true, y_pred):
    pass