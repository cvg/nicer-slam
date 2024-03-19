# adapted from https://github.com/EPFL-VILAB/omnidata and https://github.com/autonomousvision/monosdf
import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys
from tqdm import tqdm
import lzma

parser = argparse.ArgumentParser(description="Extract monocular cues from images.")

parser.add_argument("--omnidata_path", dest="omnidata_path", help="path to omnidata model")
parser.set_defaults(omnidata_path="3rdparty/omnidata/omnidata_tools/torch/")

parser.add_argument("--pretrained_models", dest="pretrained_models", help="path to pretrained models")
parser.set_defaults(pretrained_models="3rdparty/omnidata/omnidata_tools/torch/pretrained_models/")

parser.add_argument("--task", dest="task", help="normal or depth")
parser.set_defaults(task="NONE")

parser.add_argument("--img_path", dest="img_path", help="path to rgb image")
parser.set_defaults(im_name="NONE")

parser.add_argument("--output_path", dest="output_path", help="path to where output image should be stored")
parser.set_defaults(store_name="NONE")

parser.add_argument("--vis", action="store_true", help="include visualization of mono depth/normal")

args = parser.parse_args()

root_dir = args.pretrained_models
omnidata_path = args.omnidata_path

sys.path.append(args.omnidata_path)
print(sys.path)
from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform

trans_topil = transforms.ToPILImage()
os.system(f"mkdir -p {args.output_path}")
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.task == "normal":
    image_size = 384

    ## Version 1 model
    # pretrained_weights_path = root_dir + 'omnidata_unet_normal_v1.pth'
    # model = UNet(in_channels=3, out_channels=3)
    # checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

    # if 'state_dict' in checkpoint:
    #     state_dict = {}
    #     for k, v in checkpoint['state_dict'].items():
    #         state_dict[k.replace('model.', '')] = v
    # else:
    #     state_dict = checkpoint

    pretrained_weights_path = root_dir + "omnidata_dpt_normal_v2.ckpt"
    model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(p=1.0),
            transforms.Resize((image_size, image_size), interpolation=PIL.Image.BILINEAR),
            #  transforms.RandomHorizontalFlip(p=1.0),
            get_transform("rgb", image_size=None),
        ]
    )

elif args.task == "depth":
    image_size = 384
    pretrained_weights_path = root_dir + "omnidata_dpt_depth_v2.ckpt"  # 'omnidata_dpt_depth_v1.ckpt'
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    model = DPTDepthModel(backbone="vitb_rn50_384")  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=PIL.Image.BILINEAR),
            # transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

trans_rgb = transforms.Compose(
    [
        transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
        transforms.CenterCrop(image_size),
    ]
)


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)) : int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_file_name):
    with torch.no_grad():
        save_path = os.path.join(args.output_path, f"{output_file_name}_{args.task}.npy")

        # print(f'Reading input {img_path} ...')
        img = Image.open(img_path)
        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)
        # if gray image
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat(1, 3, 1, 1)

        H, W = img.height, img.width

        output = model(img_tensor).clamp(min=0, max=1)
        if args.task == "depth":
            output = F.interpolate(output.unsqueeze(0), (H, W), mode="bicubic").squeeze(0)
            output = output.clamp(0, 1)
            with lzma.open(save_path, "wb") as f:
                np.save(f, output.detach().cpu().squeeze())

            # for visualization
            if args.vis:
                plt.imsave(save_path + ".png", output.detach().cpu().squeeze(), cmap="plasma")

        else:
            trans = transforms.Compose(
                [transforms.ToPILImage(), transforms.Resize((H, W), interpolation=PIL.Image.BILINEAR)]
            )
            output = trans(output[0])
            output = np.array(output.getdata()).reshape(output.height, output.width, 3).astype(np.float32)
            output /= 255.0
            output = output.transpose(2, 0, 1)
            with lzma.open(save_path, "wb") as f:
                np.save(f, output)

            # for visualization
            if args.vis:
                output = output.transpose(1, 2, 0)
                plt.imsave(save_path + ".png", output)

        # print(f'Writing output {save_path} ...')


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for i, f in tqdm(
        enumerate(sorted(glob.glob(args.img_path + "/*_rgb.png"), key=lambda x: int(os.path.basename(x)[:-8]))[:])
    ):
        save_outputs(f, f"{i:06d}")
else:
    print("invalid file path!")
    sys.exit()
