import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from models import Generator, Discriminator
from loss import GeneratorLoss
from train_utils import train
from preprocess import TrainDatasetFromFolder
from plot_loss import plot
from evaluate import evaluate

import warnings
warnings.filterwarnings("ignore")

UPSCALE_FACTOR = 4
CROP_SIZE = 180
torch.autograd.set_detect_anomaly(True)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = TrainDatasetFromFolder("DIV2K_train_HR", crop_size=CROP_SIZE,
                                   upscale_factor=UPSCALE_FACTOR)
trainloader = DataLoader(train_set, batch_size=4, num_workers=4, shuffle=True)

netG = Generator(UPSCALE_FACTOR)
netD = Discriminator()

generator_criterion = GeneratorLoss()

generator_criterion = generator_criterion.to(device)
netG = netG.to(device)
netD = netD.to(device)

optimizerG = optim.Adam(netG.parameters(), lr=0.0002)
optimizerD = optim.Adam(netD.parameters(), lr=0.0002)

num_epochs = 50

netG, netD, g_loss, d_loss = train(trainloader, num_epochs, netG, netD, optimizerG, optimizerD, generator_criterion, device)
plot(num_epochs, g_loss, d_loss)
evaluate(netG, train_set, device)