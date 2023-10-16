import numpy as np
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt

weight_path=r'D:\\dl_project\\dl_project_cnn\\test_0.7854910683151273.pth'
new_path = r'D:\\dl_project\\dl_project_cnn\\test_modified.pth'
pth_file = torch.load(f=weight_path)
for name in list(pth_file):
    if 'out' in name:
        newname = name.strip('classifier.')
        pth_file[newname] = pth_file.pop(name)

torch.save(pth_file, f=new_path)

# for name in pth_file:
#     print(name)
