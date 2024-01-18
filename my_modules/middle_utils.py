import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import torchvision.utils as vutils

"""おそらく、0要素目はos.getcwd()を想定しているプログラム"""
def make_dir(path):
    path_list = path.split('/')
    now_path = ""
    for i, dir in enumerate(path_list):
        if i == 0:
            continue
        else:
            now_path += f"/{dir}"
            if not os.path.exists(now_path):
                os.makedirs(now_path)

class Save():
    def __init__(self, dataset):
        now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        # self追加等
        self.parent_path = os.path.join(os.getcwd(), f'middle_gene_{now}')
        self.model_path = os.path.join(self.parent_path, 'models')
        make_dir(self.model_path)

        """
        self.sample_path = os.path.join(parent_path, 'samples')
        self.tensor_path = os.path.join(parent_path, 'tensor')
        make_dir(self.sample_path)
        make_dir(self.tensor_path)
        """
   
    def save_model(self, epoch, model):
        file_nm = os.path.join(
            self.model_path, 
            f'epoch{epoch}_model.pt'
        )
        torch.save(model, file_nm)

    """大きく変更"""
    def save_img(self, epoch, sample, transit, attempt):

        if epoch == 'only_sampling':
            
            ij_parent_path = os.path.join(self.parent_path, f'transit_{transit}', f'attempt_{attempt}')
            make_dir(ij_parent_path)

            file_nm = os.path.join(ij_parent_path, 'sample.png')
        """
        else:
            file_nm = os.path.join(self.sample_path, f'epoch{epoch}_sample.png')
        """

        reverse_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1.) / 2)
        ])

        
        sample = reverse_transform(sample).cpu()
        self.save_tensor(epoch, sample, ij_parent_path)
        sample = torch.clamp(sample * 255, min=0, max=255).to(dtype=torch.uint8)
        """変更箇所"""
        grid = vutils.make_grid(sample, nrow=10, padding=2)
        fig, ax = plt.subplots(figsize=(10,100))
        ax.axis('off')
        ax.imshow(grid.permute(1, 2, 0))
        fig.savefig(file_nm, bbox_inches='tight')
        plt.close(fig)

    """i, j 追加"""
    def save_tensor(self, epoch, sample, ij_parent_path):
        if epoch == 'only_sampling':

            file_nm = os.path.join(ij_parent_path, 'tensor.pt')
        """
        else:
            file_nm = os.path.join(self.tensor_path, f'epoch{epoch}_tensor.pt')
        """
        torch.save(sample, file_nm)