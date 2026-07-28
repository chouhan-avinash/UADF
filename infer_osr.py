import os
import pathlib
from typing import Optional
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import matplotlib.pyplot as plt
import numpy as np
import opensr_test
import pandas as pd
import requests
import rioxarray as rxr
import torch
import torch.nn.functional as F

path_uadf = "/eval_results" 

def downloadSR(
    model_id: str,
    dataset_id: Optional[str] = None,
    huggingface_repo: Optional[
        str
    ] = "https://huggingface.co/isp-uv-es/superIX/resolve/main",
) -> pathlib.Path:

    if dataset_id is None:
        dataset_id = opensr_test.datasets

    if isinstance(dataset_id, str):
        dataset_id = [dataset_id]

    for db in dataset_id:
        # download & load metadata
        print(f"Downloading {db}")
        metadata = f"https://huggingface.co/datasets/isp-uv-es/opensr-test/resolve/main/100/{db}/{db}_metadata.csv"
        metadata_db = pd.read_csv(metadata)

        # Set the model output path
        model_outpath = pathlib.Path(f"{model_id}/results/SR/{db}/geotiff")
        model_outpath.mkdir(parents=True, exist_ok=True)

        for i, row in metadata_db.iterrows():
            # File to download
            hr_file = row["hr_file"]
            dataset_path = (
                f"{huggingface_repo}/{model_id}/results/SR/{db}/geotiff/{hr_file}.tif"
            )

            # Download the file
            with requests.get(dataset_path, stream=True) as r:
                r.raise_for_status()
                with open(model_outpath / hr_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

    return pathlib.Path(f"{model_id}")





def image_resize(image: np.ndarray, size: int) -> np.ndarray:

    image = torch.from_numpy(image)
    '''image = (
        torch.nn.functional.interpolate(
            image / 10000, size=size, mode="bilinear", antialias=True
        )
        * 10000
    )'''
    image = (
        torch.nn.functional.interpolate(
            image / 10000, size=size, mode="bilinear", antialias=True
        )
        * 10000 )   
    return image.squeeze().numpy() 


def image_resize_sr(image: np.ndarray, size: int) -> np.ndarray:

    image = torch.from_numpy(image)
    image = (
        torch.nn.functional.interpolate(
            image / 10000, size=size, mode="bilinear", antialias=True
        )
        * 10000
    )
    return image.squeeze().numpy()
def image_resize_sr_1(image: np.ndarray, size: int) -> np.ndarray:

    image = torch.from_numpy(image)
    image = (
        torch.nn.functional.interpolate(
            image , size=size, mode="bilinear", antialias=True
        )
        
    )
    return image.squeeze().numpy()    

    
import numpy as np
from PIL import Image
from pathlib import Path

import numpy as np
from PIL import Image

import numpy as np
from PIL import Image

def save_png(img, path):

    # torch -> numpy
    if hasattr(img, "cpu"):
        img = img.detach().cpu().numpy()

    #print("before:", img.shape)

    # remove batch dim
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    # grayscale HWC -> HW
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]

    #print("after:", img.shape)

    # [0,1] -> uint8
    img = (img * 255.0).clip(0,255).astype(np.uint8)

    Image.fromarray(img).save(path)
    
import numpy as np
import rasterio

def save_tif(img, path):

    # torch -> numpy
    if hasattr(img, "cpu"):
        img = img.detach().cpu().numpy()

    # remove batch dim
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    # expected CHW
    # shape: (C,H,W)

    # [0,1] -> [0,10000]
    img = (img * 10000.0).clip(0, 10000).astype(np.uint16)

    c, h, w = img.shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=img.dtype,
    ) as dst:

        for i in range(c):
            dst.write(img[i], i + 1)    
def run(diffusion,
    model_id: str,
    sr_in_bands: list,
    name_csv: str,
    scale: int = 4,
    dataset_ids: Optional[str] = None,
    experiment: Optional[dict] = None,
    output_dir: Optional[str] = None,
    compute_plots: Optional[dict] = {
        "triplets": True,
        "summary": True,
        "tc": True,
        "histogram": True,
        "ternary": True,
    },
) -> None:

    exp_object = opensr_test.Metrics(**experiment)
    dmetric = experiment["correctness_distance"]
    columns = [
        "model",
        "dataset",
        "reflectance",
        "spectral",
        "spatial",
        "synthesis",
        "ha_metric",
        "om_metric",
        "im_metric",
    ]
    condition = (dmetric == "clip") or (dmetric == "lpips")
    df = pd.DataFrame(columns=columns)

    for dataset in dataset_ids:
        # Set the output directory
        if output_dir is None:
            output_dir = pathlib.Path(f"{model_id}/results/SR/{dataset}/figures/")
            output_dir.mkdir(parents=True, exist_ok=True)

        output_dir_triplets = output_dir / "triplets"
        output_dir_triplets.mkdir(parents=True, exist_ok=True)

        output_dir_summary = output_dir / f"summary_{dmetric}"
        output_dir_summary.mkdir(parents=True, exist_ok=True)

        output_dir_tc = output_dir / f"triplets_tc_{dmetric}"
        output_dir_tc.mkdir(parents=True, exist_ok=True)

        output_dir_histogram = output_dir / f"histogram_{dmetric}"
        output_dir_histogram.mkdir(parents=True, exist_ok=True)

        output_dir_ternary = output_dir / f"ternary_{dmetric}"
        output_dir_ternary.mkdir(parents=True, exist_ok=True)


        data = opensr_test.load(dataset, version = "v2")
        lr, hr = data["L2A"][:, [3, 2, 1]], data["HRharm"][:, 0:3]
        metadata = data["metadata"]


        hr_scale = hr.shape[2] // lr.shape[2]


        for i in range(len(lr)):
            print(f"Processing {dataset}: image {i + 1}/{len(lr) + 1}")
            # Load image by image
            lr_img, hr_img = lr[i], hr[i], 
            lr_img = torch.from_numpy(lr_img) / 10000 #10000
            hr_img = torch.from_numpy(hr_img) / 10000 #10000

            
  
            diffusion.feed_data_opensr1(lr_img.unsqueeze(0))
            diffusion.test_osr_nw2_full(continous=False)  #test_osr
            visuals = diffusion.get_current_visuals_opensr1(need_LR=True)
            sr_img = visuals['SR'].squeeze(0) #.clamp_(0.0,1.0)
            sr_img1 = visuals['SR2'].squeeze(0)
            if dataset == "venus":
                  sr_img = F.interpolate(
                  sr_img.unsqueeze(0),
                  scale_factor=0.5,
                  mode="bilinear",
                  antialias=True
                  ).squeeze(0)

          

            
            if condition:

                lr_img = (lr_img * 3).clip(0, 1)
                hr_img = (hr_img * 3).clip(0, 1)
                sr_img = (sr_img * 3).clip(0, 1)

            
            results = exp_object.compute(lr_img.float(), sr_img.float(), hr_img.float())
            results["model"] = model_id
            results["dataset"] = dataset
            df.loc[len(df)] = results

            # Save the figures
            if compute_plots["triplets"]:
                fig, axs = exp_object.plot_triplets()
                fig.savefig(output_dir_triplets / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["summary"]:
                fig, axs = exp_object.plot_summary()
                fig.savefig(output_dir_summary / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["tc"]:
                fig, axs = exp_object.plot_tc()
                fig.savefig(output_dir_tc / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["histogram"]:
                fig, axs = exp_object.plot_histogram()
                fig.savefig(output_dir_histogram / f"{dataset}__{hr_name}.png")
                plt.close(fig)

            if compute_plots["ternary"]:
                fig, axs = exp_object.plot_ternary()
                fig.savefig(output_dir_ternary / f"{dataset}__{hr_name}.png")
                plt.close(fig)



    # ---- Compute Average Metrics ----
    numeric_cols = [
        "reflectance",
        "spectral",
        "spatial",
        "synthesis",
        "ha_metric",
        "om_metric",
        "im_metric",
    ]

    avg_row = df[numeric_cols].mean().to_dict()
    avg_row["model"] = model_id
    avg_row["dataset"] = "AVERAGE"

    df.loc[len(df)] = avg_row




    # Save the results
    df.to_csv(output_dir.parent / name_csv, index=False)

    return df


datasets = ["naip", "spot", "spain_crops", "spain_urban"] #,"venus" , , "spot", "spain_crops", "spain_urban","venus"]


experiment_01 = {
    "device": "cuda",
    "agg_method": "patch",
    "patch_size": 1,
    "correctness_distance": "nd",
    "border_mask": 64,
}



experiment_02 = {
    "device": "cuda",
    "agg_method": "patch",
    "patch_size": 16,
    "correctness_distance": "lpips",
    "border_mask": 64,
}
experiment_03 = {
    "device": "cuda",
    "agg_method": "patch",
    "patch_size": 16,
    "correctness_distance": "clip",
    "border_mask": 64,
} 
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/sr6_1.json',
                        help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
parser.add_argument('-gpu', '--gpu_ids', type=str, default='6')
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-enable_wandb', action='store_true')
parser.add_argument('-log_infer', action='store_true')

# parse configs
args = parser.parse_args()
opt = Logger.parse(args)

opt = Logger.dict_to_nonedict(opt)
diffusion = Model.create_model(opt)
#logger.info('Initial Model Finished')

diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
print("===========", path_uadf)
results_1 = run(diffusion,
    model_id=path_uadf,
    sr_in_bands=[0, 1, 2],
    dataset_ids=datasets,
    experiment=experiment_02,
    name_csv="results_lpips.csv",
    compute_plots={
        "triplets": True,
        "summary": True,
        "tc": True,
        "histogram": True,
        "ternary": True,
    },
)


results_1 = run(diffusion,
    model_id=path_uadf,
    sr_in_bands=[0, 1, 2],
    dataset_ids=datasets,
    experiment=experiment_03,
    name_csv="results_clip.csv",
    compute_plots={
        "triplets": True,
        "summary": True,
        "tc": True,
        "histogram": True,
        "ternary": True,
    },
)

results_1 = run(diffusion, 
    model_id=path_uadf,
    sr_in_bands=[0, 1, 2],
    dataset_ids=datasets,
    experiment=experiment_01,
    name_csv="results_nd.csv",
    compute_plots={
        "triplets": True,
        "summary": True,
        "tc": True,
        "histogram": True,
        "ternary": True,
    },
)




