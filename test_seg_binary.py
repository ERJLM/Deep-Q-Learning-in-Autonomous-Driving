import argparse

from torchmetrics import JaccardIndex

from models import SegModel_Binary

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model-seg-binary.pth')
parser.add_argument('--dataset', default='BDD10K_Binary')
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--imgsize', type=int, default=1024)
parser.add_argument('--nregions', type=int, default=4)
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchmetrics
from torchmetrics.classification import MulticlassJaccardIndex, BinaryJaccardIndex
from tqdm import tqdm
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################### DATA AUG ###########################

assert args.imgsize % args.nregions == 0
region_size = args.imgsize // args.nregions

transform = A.Compose([
    A.Resize(args.imgsize, args.imgsize),
    A.Normalize(0, 1),
    ToTensorV2(),
])

############################# DATA #############################

ds = getattr(data, args.dataset)
ds = ds('/Users/eliandromelo/Downloads', 'test', transform)
ts = torch.utils.data.DataLoader(ds, args.batchsize, num_workers=0, pin_memory=True)

############################ MODEL ############################

model = torch.load(args.model, map_location=device)

############################ LOOP ############################

metric = BinaryJaccardIndex().to(device)

def to_patches(x):
    k = x.shape[1]
    x = x.unfold(2, region_size, region_size).unfold(3, region_size, region_size)
    x = x.permute(0, 2, 3, 1, 4, 5)
    x = x.reshape(-1, k, region_size, region_size)
    return x

with torch.no_grad():
    c = 0
    total = 0
    correct_pos = correct_neg = 0
    total_pos = total_neg = 0
    for d in tqdm(ts):
         images = d['image'].float().to(device)  # N x C x H x W

         masks = d['mask'].float().to(device).unsqueeze(1)  # N x 1 x H x W

         images = to_patches(images)

         masks = to_patches(masks)

         preds = model(images)['out']
         preds = preds.gt(0.5)
         print(preds)

         correct_pos += torch.sum((masks == 1) & (preds == 1))
         correct_neg += torch.sum((masks == 0) & (preds == 0))
         total_pos += torch.sum(masks == 1)
         total_neg += torch.sum(masks == 0)
         metric.update(preds, masks)

         bacc = (correct_pos / total_pos + correct_neg / total_neg) / 2
         print(f'Estimated balanced accuracy so far: {100 * bacc} %')

    bacc = (correct_pos / total_pos + correct_neg / total_neg) / 2
    print(f'Balanced accuracy of the network: {100 * bacc} %')

print(args.model, metric.compute().item())
