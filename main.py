
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

if torch.cuda.is_available():
    print('cuda works')
else:
    print('no')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
