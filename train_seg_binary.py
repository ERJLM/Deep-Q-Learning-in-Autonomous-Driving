import argparse
from multiprocessing import freeze_support

from models import SegModel_Binary
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
#parser.add_argument('--model', default='model-seg-binary.pth')
parser.add_argument('--output', default='model-seg-binary.pth')
parser.add_argument('--dataset', default='BDD10K_Binary')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--imgsize', type=int, default=2048)
parser.add_argument('--nregions', type=int, default=8)
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch, torchvision
from time import time
from tqdm import tqdm
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#print(device)

########################### DATA AUG ###########################

assert args.imgsize % args.nregions == 0
region_size = args.imgsize // args.nregions

transform = A.Compose([
    A.Resize(args.imgsize, args.imgsize),
    A.RandomCrop(region_size, region_size),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(p=1),
    A.Normalize(0, 1),
    ToTensorV2(),
])




def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    ############################# DATA #############################
#torch.set_num_threads(1)  # Set the number of threads explicitly

ds = getattr(data, args.dataset)
ds = ds('/Users/eliandromelo/Downloads', 'train', transform)
tr = torch.utils.data.DataLoader(ds, args.batchsize, True, num_workers=4, pin_memory=True)

    ############################ MODEL ############################
#torch.set_num_threads(1)  # Set the number of threads explicitly

model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=ds.num_classes)
model.to(device)
#model = torch.load(args.model, map_location=device)
    ############################ LOOP ############################

opt = torch.optim.Adam(model.parameters())
loss_function = torch.nn.BCEWithLogitsLoss()
if __name__ == '__main__':
    prev_loss = float('inf')
    for epoch in range(args.epochs):
        #print("log1")
        #torch.set_num_threads(1)  # Set the number of threads explicitly
        tic = time()
        loss_avg = 0
        for d in tqdm(tr):
            #print("log2")
            image = d['image'].to(device)
            mask = d['mask'].to(device).float()
            ypred = model(image)['out']
            ypred = ypred.squeeze()
            loss = loss_function(ypred, mask)
            loss_avg += float(loss) / len(tr)
            opt.zero_grad()
            loss.backward()
            opt.step()  # w = w - eta*dloss/dw
        toc = time()
        print(f'Epoch {epoch + 1}/{args.epochs} - {toc - tic:.0f}s - Loss: {loss_avg}')
        if loss_avg < prev_loss:
            prev_loss = loss_avg
            torch.save(model, args.output)

