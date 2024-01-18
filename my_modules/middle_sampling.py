import numpy as np
import torch

from run import score_fn

class EulerMaruyamaMethod():
    def __init__(self, config):
        self.config = config

    def step(self, x, t, model, sde):
        dt = -1. / self.config['sde']['timesteps']
        z = torch.randn_like(x)
        score = score_fn(x, t, model, sde)
        drift, diffusion = sde.reverse().sde(x, t, score)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z

        return x, x_mean

"""transitを中間地点にして生成する"""
def middle_sampling(config, shape, shape_one, model, sde, transit):
    sampler = EulerMaruyamaMethod(config)
    with torch.no_grad():

        """中間地点まで１枚で生成"""

        x = sde.prior_sampling(shape_one).to(config['device'])
        timesteps = torch.linspace(
            config['sde']['T'],
            config['sde']['eps'],
            config['sde']['timesteps'],
            device=config['device']
        )

        for i,t in enumerate(timesteps):

            if i<transit:
                vec_t = torch.ones(shape_one[0], device=t.device) * t
                x, x_mean = sampler.step(x, vec_t, model, sde)


                """中間地点で全コピー"""
            elif i==transit:
                x = x.repeat(shape[0], 1, 1, 1)

            elif i>=transit:
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = sampler.step(x, vec_t, model, sde)



                

        """
        for i,t in enumerate(timesteps):
            vec_t = torch.ones(shape[0], device=t.device) * t
            x, x_mean = sampler.step(x, vec_t, model, sde)
            

            if i==transit:
                for k in range(len(x)-1):   # 全コピー
                    x[k, :, :, :] = x[-1, :, :, :]
        """

    
    return x_mean