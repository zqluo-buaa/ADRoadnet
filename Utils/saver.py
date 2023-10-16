import numpy as np
import torch
import os


class _Saver(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

class Saver(_Saver):
    """
    Save the intermediate model
    """

    def __init__(self, base_path, basis_name):
        self.base_path = base_path
        self.basis_name = basis_name

    def update(self, model, value):
        save_name = self.basis_name + str(value) + '.pth'
        save_path = os.path.join(self.base_path, save_name)

        for file in os.listdir(self.base_path):
            if self.basis_name in file:
                os.remove(os.path.abspath(file))

        torch.save(model.state_dict(), save_path)

