import torch
import numpy as np


def extract(a, t, shape):
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

import numpy as np

def make_beta_schedule(
    n_timestep=2000,
    linear_start=5e-4,
    linear_end=2e-2
):

    return np.linspace(
        linear_start,
        linear_end,
        n_timestep,
        dtype=np.float32
    )
class DDIMSampler:

    def __init__(
    self,
    denoiser,
    sampling_steps=50,
    eta=0.0,
    device="cuda"
    ):

        self.model = denoiser.eval().to(device)
        self.device = device
        betas = make_beta_schedule(
            n_timestep=2000,
            linear_start=5e-4,
            linear_end=2e-2
        )
        betas = torch.tensor(
            betas,
            dtype=torch.float32,
            device=device
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas_cumprod = alphas_cumprod

        self.sqrt_recip_alphas_cumprod = torch.sqrt(
            1.0 / alphas_cumprod
        )

        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / alphas_cumprod - 1
        )

        self.num_timesteps = 2000
        self.sampling_steps = sampling_steps
        self.eta = eta

    def predict_start_from_noise(
        self,
        xt,
        t,
        noise
    ):

        return (
            extract(
                self.sqrt_recip_alphas_cumprod,
                t,
                xt.shape
            ) * xt
            -
            extract(
                self.sqrt_recipm1_alphas_cumprod,
                t,
                xt.shape
            ) * noise
        )

    @torch.no_grad()
    def sample(self, lr):

        b = lr.shape[0]

        x = torch.randn_like(lr)

        times = torch.linspace(
            -1,
            self.num_timesteps - 1,
            steps=self.sampling_steps + 1,
            device=self.device
        ).int()

        times = list(reversed(times.tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:

            t = torch.full(
                (b,),
                time,
                device=self.device,
                dtype=torch.long
            )

            # Predict noise using TorchScript denoiser
            pred_noise = self.model(
    x,
    lr,
    t
            )

            # Estimate x0
            x_start = self.predict_start_from_noise(
    x,
    t,
    pred_noise
            )

            # Clip x0
            x_start.clamp_(-1., 1.)

            # Recompute predicted noise from the clipped x0
            pred_noise = self.predict_noise_from_start(
    x,
    t,
    x_start
            )

            if time_next < 0:
                x = x_start
                break

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.eta * (
                (1 - alpha / alpha_next)
                * (1 - alpha_next)
                / (1 - alpha)
            ).sqrt()

            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(x)

            x = (
                x_start * alpha_next.sqrt()
                + c * pred_noise
                + sigma * noise
            )

        return x
        
    def predict_noise_from_start(self, x_t, t, x0):

        return (
        (
            extract(
                self.sqrt_recip_alphas_cumprod,
                t,
                x_t.shape
            ) * x_t
            - x0
        ) /
        extract(
            self.sqrt_recipm1_alphas_cumprod,
            t,
            x_t.shape
        )
        )        
        
