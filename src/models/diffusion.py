import torch
from torch import nn
from diffusers import DDPMScheduler, DDIMScheduler
from .net import Genet

class DiffusionModel(nn.Module):
    """
    For scRNA perturbation prediction,
    Implement diffusion on perturbed cells Y, conditioned on control cells X and embeddings of perturbed genes 
    """

    def __init__(self, configs):
        super(DiffusionModel, self).__init__()
        self.device = configs["training"]["device"]
        self.diffsteps = configs["model"]["diffusion_steps"]
        self.batch_size = configs["training"]["batch_size"]
        self.net = Genet(configs).to(self.device)
        # DDPM/ DDIM scheduler
        self.ddpm_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear')
        self.ddim_scheduler = DDIMScheduler(num_train_timesteps=self.diffsteps, beta_schedule='scaled_linear', timestep_spacing="trailing")

    def forward(self, x, y, embeddings):
        noise = torch.randn_like(y)    # Random Gaussian noise
        timesteps = torch.linspace(0,999, self.batch_size).long().to(self.device)
        noised = self.ddpm_scheduler.add_noise(y, noise, timesteps)  # Add noise
        pred = self.net(noised, x, embeddings, timesteps)
        return pred, noise

    def reverse(self, x, embeddings):
        x = x.to(self.device).float()
        embeddings = embeddings.to(self.device).float()
        with torch.no_grad():
            x_t = torch.randn_like(x).to(self.device)
            if self.diffsteps < 1000:
                scheduler = self.ddim_scheduler
            else:
                scheduler = self.ddpm_scheduler
            scheduler.set_timesteps(self.diffsteps)
            scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)
            for t in scheduler.timesteps:
                t_tensor = torch.full((self.batch_size,), t, dtype=torch.long, device=self.device)
                t = torch.tensor([t], dtype=torch.long)
                pred_noise = self.net(x_t, x, embeddings, t_tensor)
                step_result = scheduler.step(
                    model_output=pred_noise,           
                    timestep=t,    
                    sample=x_t                   
                )
                x_t = step_result.prev_sample
            y_pred = x_t
        return y_pred

if __name__ == '__main__':
    pass