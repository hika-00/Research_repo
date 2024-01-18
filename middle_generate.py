
import os
import sys
import argparse

import torch
import torch.nn as nn


"""moduleのパス追加 カレントディレクトリ以外も"""
submodule_path = os.path.join(os.getcwd(), 'my-submodule')
mymodules_path = os.path.join(os.getcwd(), 'my_modules')
sys.path.append(submodule_path)
sys.path.append(mymodules_path)

from configs.generate_config import load_config_generate
from configs.main_config import load_config
from configs.cifar10_config import load_config_cifar10
from configs.food_101_config import load_config_food_101
from configs.food_101_small_config import load_config_food_101_small
from model import UNet
from middle_sampling import middle_sampling   # 変更箇所
from sde import VPSDE
from middle_utils import Save   # 変更箇所



parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    help="Select the dataset from ['food-101', 'food-101-small', 'cifar10']"
)
"""ipynb用 cifar10固定"""
args = parser.parse_args(args=['cifar10'])


if __name__ == '__main__':
    config = load_config()
    generate_config = load_config_generate(args.dataset.lower())
    if args.dataset.lower() == 'cifar10':
        config['data'] = load_config_cifar10()
    elif args.dataset.lower() == 'food-101':
        config['data'] = load_config_food_101()
    elif args.dataset.lower() == 'food-101-small':
        config['data'] = load_config_food_101_small()
    else:
        raise NotImplementedError("Dataset is not supported.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device

    model = torch.load(generate_config['model_path'])
    model = model.to(config['device']).to(dtype=torch.float32)
    model = nn.DataParallel(model)
    SDE = VPSDE(config)
    save = Save(args.dataset)

    """middle_generate用なのでこのファイルでのみ変更"""
    config['sampling']['n_img'] = 400
    shape = (
                config['sampling']['n_img'], config['data']['channel'],
                config['data']['height'], config['data']['width']
            )
    
    shape_one = (
                1, config['data']['channel'],
                config['data']['height'], config['data']['width']
            )
    
    epoch = 'only_sampling'
    save.save_model(epoch, model)

    """transitはReverseプロセスの回数 Reverse process の t の定義は1→0"""
    # transit = [650, 700, 750, 800, 850, 900, 950]
    transit = [600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50]
    attempt = 30

    for i in transit:
        print(f'{i=}')
        for j in range(attempt):
            print(f'{j=}')
            
            sample = middle_sampling(config, shape, shape_one, model, SDE, i) 
            save.save_img(epoch, sample, i, j)