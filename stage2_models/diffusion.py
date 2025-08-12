import math
import torch
import torch.nn.functional as F

from torch import nn
from collections import namedtuple
from stage2_models.denoiser import Denoiser

__all__ = ["Diffusion"]
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(self, args):
        super(GaussianDiffusion, self).__init__()
        self.args = args

        timesteps = args["Diffusion"].get('timesteps', 1000)
        sampling_timesteps = args["Diffusion"].get('ddim_timesteps', 5)
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = args["Diffusion"].get('scale', 1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',(1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.Denoiser = Denoiser(num_layers=args["Diffusion"]["Denoiser"].get('num_layers', 2),
                                 num_heads=args["Diffusion"]["Denoiser"].get('num_heads', 4),
                                 embedding_dim=args["Diffusion"]["Denoiser"].get('embedding_dim', 512),
                                 hidden_size=args["Diffusion"]["Denoiser"].get('hidden_size', 512),
                                 ff_size=args["Diffusion"]["Denoiser"].get('ff_size', 2048),
                                 dropout=args["Diffusion"]["Denoiser"].get('dropout', 0.1))

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, c, t, mask):
        x_t = x / self.scale
        pred_pose = self.Denoiser(x=x_t, c=c, t=t, mask=mask)
        x_start = pred_pose
        x_start = x_start * self.scale
        pred_noise = self.predict_noise_from_start(x_t, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(self, x, c, mask):
        batch = x.shape[0]
        shape = (batch, x.shape[1], self.args["Diffusion"]["Denoiser"].get('hidden_size', 512))
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) 

        img = torch.randn(shape, device=x.device)
        x_start = None
        preds_all=[]

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=x.device, dtype=torch.long)

            preds = self.model_predictions(x=img, c=c, t=time_cond, mask=mask)
            pred_noise, x_start = preds.pred_noise.float(), preds.pred_x_start
            preds_all.append(x_start)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
        return preds_all

    def prepare_targets(self, targets):
        diffused_poses = []
        noises = []
        ts = []
        for i in range(0,targets.shape[0]):
            targets_per_sample = targets[i]

            d_poses, d_noise, d_t = self.prepare_diffusion_concat(targets_per_sample)
            diffused_poses.append(d_poses)
            noises.append(d_noise)
            ts.append(d_t)

        return torch.stack(diffused_poses), torch.stack(noises), torch.stack(ts)

    def prepare_diffusion_concat(self, point_feats):

        t = torch.randint(0, self.num_timesteps, (1,), device='cuda').long()
        noise = torch.randn(point_feats.shape[0], 512, device='cuda')
        x_start = point_feats
        x_start = x_start * self.scale
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x = x / self.scale

        return x, noise, t

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x, c, mask, is_train):
        if not is_train:
            results = self.ddim_sample(x=x, c=c, mask=mask)
            return results[self.sampling_timesteps-1]

        if is_train:
            x_t, noises, t = self.prepare_targets(x)
            x_t = x_t.float()
            t = t.squeeze(-1)
            preds = self.Denoiser(x=x_t, c=c, mask=mask, t=t)
            return preds
