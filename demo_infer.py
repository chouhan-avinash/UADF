"""
OpenSR Demo Inference

Author:
    Dr. Avinash Chouhan
    North Eastern Space Applications Centre (NESAC)

GitHub:
    https://github.com/chouhan-avinash/UADF
    
Copyright (c) 2026
Licensed under the MIT License.
"""

import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
from dif_wrapper import DDIMSampler

device = "cuda"



denoiser = torch.jit.load(
    "/uadf_export/osr_denoiser_512.pt",
    map_location=device
)


test_new = torch.jit.load("/uadf_export/osr_denoiser_step1_512.pt")
print(denoiser)
sampler = DDIMSampler(
    denoiser,
    sampling_steps=100,
    eta=0.0,
    device=device
)

############################################

import rioxarray as rxr
import torch

# Read GeoTIFF
img = rxr.open_rasterio("/lr/LR__ROI_00026__20200730T110619_20200730T111530_T30TUM.tif").values.astype("float32")
img = img[[3, 2, 1]]

tensor = torch.from_numpy(img) / 10000.0


tensor = tensor.unsqueeze(0).to(device)
tensor = F.interpolate(
    tensor,
    scale_factor=4,
    mode="bilinear",
    align_corners=False
)


with torch.no_grad():
    tensor = test_new(tensor)
    sr = sampler.sample(tensor)

############################################


sr = sr.clamp(0, 1)

T.ToPILImage()(sr[0].cpu()).save("sr_512_both.png")
