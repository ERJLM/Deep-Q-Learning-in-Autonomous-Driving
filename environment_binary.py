import torch
from models import SegModel
import argparse
from multiprocessing import freeze_support
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model-binary-1024-4.pth')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_patches(x, region_size):  # x.shape = (N, C, H, W)
    k = x.shape[1]
    x = x.unfold(2, region_size, region_size).unfold(3, region_size, region_size)
    x = x.permute(0, 2, 3, 1, 4, 5)
    x = x.reshape(-1, k, region_size, region_size)
    return x


class Environment:
    def __init__(self, seg_model, nregions, device):
        self.seg_model = torch.load(seg_model, map_location=device)
        self.seg_model.eval()
        self.seg_model.backbone.register_forward_hook(self._latent_space_hook)
        self.device = device
        self.nregions = nregions

    def _latent_space_hook(self, module, args, output):
        if isinstance(output, dict) and 'out' in output:
            self.latent = output['out'].mean([2, 3]).squeeze().to(device)  # (1, 2048, 32, 32) -> (1, 2048)
        else:
            # Handle the case where 'out' is not present in the output or output is not a dictionary
            # You may print or log a message to help with debugging
            # print("Invalid output format in _latent_space_hook:", output)
            # Set self.latent to an appropriate default value or handle the error accordingly
            self.latent = torch.zeros((1, 1024), device=self.device)

    def _calc_reward(self, mask):
        return (mask == 1).float().mean().to(device)/ self.iteration
        # should we divide by the number of iterations?

    def reset(self, image, mask):
        region_size = image.shape[1] // self.nregions
        self.image_regions = to_patches(image[None], region_size)
        self.image_regions = self.image_regions.to(device)
        self.mask = mask.long()[None, None].to(device)
        self.state = torch.zeros((1, 1024, self.nregions, self.nregions), device=self.device) #image
        # print(f"State_length: {self.state.shape}")
        self.region_selected = torch.zeros(self.nregions ** 2, dtype=bool, device=self.device)
        self.iteration = 0
        self.score = torch.zeros_like((self.mask).float().mean()).to(device)
        self.output = torch.zeros((1, 4, image.shape[1], image.shape[2]), dtype=torch.int32, device=device)

    def possible_actions(self):
        return self.nregions ** 2

    def was_region_selected(self, action):
        return self.region_selected[action]

    def get_state(self):
        return self.state

    def step(self, action):
       # print(action, self.possible_actions())
        assert 0 <= action < self.possible_actions()
        self.iteration += 1
        self.score += (self.output == self.mask).float().mean()

        if self.was_region_selected(action):
            return -10, False, self.score
        self.region_selected[action] = True
        with torch.no_grad():
            preds = self.seg_model(self.image_regions[[action]])['out']  # (1, 1, 256, 256)
            preds = torch.sigmoid(preds) >= 0.5
        x = action % self.nregions
        y = action // self.nregions
        self.state[..., y, x] = self.latent[0]
        reward = self._calc_reward(preds)
        #self.score += reward
        isover = torch.all(self.region_selected)

        # How many times he can iterate to choose all the zones
        if self.iteration > 32:
            print("Number of iterations:", self.iteration)
            print("Reward:", reward)
            print("Regions selected", self.region_selected.sum().item())
            return 0, True, self.score
        self.output[:, :, y * preds.shape[2]:(y + 1) * preds.shape[2],
        x * preds.shape[3]:(x + 1) * preds.shape[3]] = preds

        if isover:
            print("Number of iterations:", self.iteration)
            print("Reward:", reward)
            print("Regions selected", self.region_selected.sum().item())
        return reward, isover, self.score


if __name__ == '__main__':  # DEBUG
    from skimage.io import imread
    from skimage.transform import resize

    image_path = '/Users/eliandromelo/Downloads/bdd100k/images/10k/train/0a0eaeaf-9ad0c6dd.jpg'
    mask_path = '/Users/eliandromelo/Downloads/bdd100k/labels/sem_seg/masks/train/0a0eaeaf-9ad0c6dd.png'
    from torchvision import models

    # Load a pre-trained DeepLabV3 model with ResNet50 backbone
    # model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model = torch.load(args.model, map_location=torch.device('cpu')).to(device)

    image = resize(imread(image_path), (1024, 1024))
    mask = resize(imread(mask_path), (1024, 1024))
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    mask = torch.tensor(mask)
    env = Environment(args.model, 4, device)
    env.reset(image, mask)
    import matplotlib.pyplot as plt
    for i in range(16):
      plt.imshow(env.image_regions[i].permute(1, 2, 0))
      plt.savefig(f'debug-{i}.jpg')
    #print('possible actions:', env.possible_actions())
    #print('was region selected:', env.was_region_selected(4))
    #print('get state:', env.get_state().shape)
    for i in range(16):
        print('step:', i, env.step(i))
