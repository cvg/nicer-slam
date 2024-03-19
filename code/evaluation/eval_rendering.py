import os
import sys

sys.path.append("../code")
import torch
import random
import argparse
import numpy as np
import utils.plots as plt
import utils.general as utils
from utils import rend_util
from training.volsdf_train import SLAMRunner

import utils.SSIM as SSIM
import lpips
from PIL import Image
import pandas as pd

ssim_computer = SSIM.SSIM()
lpips_computer = lpips.LPIPS(net="alex").cuda()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./confs/replica/runconf_replica_2.conf")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument(
        "--is_continue", default=False, action="store_true", help="If set, indicates continuing from a previous run."
    )
    parser.add_argument(
        "--new_expfolder",
        default=False,
        action="store_true",
        help="If set, create new expfolder when continuing from a previous run.",
    )
    parser.add_argument(
        "--timestamp",
        default="latest",
        type=str,
        help="The timestamp of the run to be used in case of continuing from a previous run.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        type=str,
        help="The checkpoint of the run to be used in case of continuing from a previous run.",
    )
    parser.add_argument(
        "--scan_id", type=int, default=-1, help="If set, taken to be the scan id. Overrides the conf file."
    )
    parser.add_argument(
        "--eval_method", default="extrapolate", type=str, help="eval method, extrapolate or interpolate"
    )
    opt = parser.parse_args()

    evalrunner = SLAMRunner(
        conf=opt.conf,
        expname=opt.expname,
        exps_folder_name=opt.exps_folder,
        is_continue=opt.is_continue,
        timestamp=opt.timestamp,
        new_expfolder=opt.new_expfolder,
        checkpoint=opt.checkpoint,
        scan_id=opt.scan_id,
    )

    dataset_conf = evalrunner.conf.get_config("dataset")
    if opt.eval_method == "extrapolate":
        dataset_conf["data_dir"] += "_EVAL_EXT"
        dataset_conf["n_images"] = 100

    evalrunner.data_dir = dataset_conf["data_dir"]
    evalrunner.eval_method = opt.eval_method

    evalrunner.train_dataset = utils.get_class(evalrunner.conf.get_string("train.dataset_class") + "_EVAL")(
        checkpoints_path=evalrunner.checkpoints_path, eval_method=opt.eval_method, **dataset_conf
    )

    evalrunner.eval_dataloader = torch.utils.data.DataLoader(
        evalrunner.train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=evalrunner.train_dataset.collate_fn,
        pin_memory=True,
    )

    evalrunner.evaldir = os.path.join(evalrunner.expdir, evalrunner.timestamp, "eval_rendering")
    utils.mkdir_ifnotexists(evalrunner.evaldir)

    images_dir = "{0}/rendering".format(evalrunner.evaldir)
    if opt.eval_method == "extrapolate":
        images_dir += "_extrapolation"
    else:
        images_dir += "_interpolation"

    utils.mkdir_ifnotexists(images_dir)

    psnrs = []
    ssims = []
    lpipss = []
    for data_index, (indices, model_input, ground_truth) in enumerate(evalrunner.eval_dataloader):
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["pose"] = model_input["pose"].cuda()
        rgb_gt = ground_truth["rgb"]
        rgb_gt = plt.lin2img(rgb_gt, evalrunner.img_res).detach().cpu().numpy()[0]
        rgb_gt = rgb_gt.transpose(1, 2, 0)
        img = Image.fromarray((rgb_gt * 255).astype(np.uint8))
        img.save("{0}/gt_{1}.png".format(images_dir, "%04d" % indices[0]))

        split = utils.split_input(model_input, evalrunner.total_pixels, n_pixels=evalrunner.split_n_pixels)
        res = []
        for s in split:
            out = evalrunner.model(s, indices, ground_truth, mode="mapping_vis")
            res.append(
                {
                    "rgb_values": out["rgb_values"].detach(),
                }
            )

        batch_size = ground_truth["rgb"].shape[0]
        model_outputs = utils.merge_output(res, evalrunner.total_pixels, batch_size)
        rgb_eval = model_outputs["rgb_values"]
        rgb_eval = rgb_eval.reshape(batch_size, evalrunner.total_pixels, 3)
        rgb_eval = plt.lin2img(rgb_eval, evalrunner.img_res).detach().cpu().numpy()[0]
        rgb_eval = rgb_eval.transpose(1, 2, 0)
        img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
        img.save("{0}/eval_{1}.png".format(images_dir, "%04d" % indices[0]))

        residual = np.abs(rgb_gt - rgb_eval)
        img = Image.fromarray((residual * 255).astype(np.uint8))
        img.save("{0}/residual_{1}.png".format(images_dir, "%04d" % indices[0]))

        psnr = rend_util.get_psnr(model_outputs["rgb_values"], ground_truth["rgb"].cuda().reshape(-1, 3)).item()
        ssim = rend_util.get_ssim(
            model_outputs["rgb_values"], ground_truth["rgb"].cuda().reshape(-1, 3), evalrunner.img_res, ssim_computer
        ).item()

        lpips = rend_util.get_lpips(
            model_outputs["rgb_values"], ground_truth["rgb"].cuda().reshape(-1, 3), evalrunner.img_res, lpips_computer
        ).item()
        print(psnr, ssim, lpips)
        psnrs.append(psnr)
        ssims.append(ssim)
        lpipss.append(lpips)

    psnrs = np.array(psnrs).astype(np.float64)
    PSNR_MSG = "psnr mean = {0} ; psnr std = {1}".format(
        "%.2f" % psnrs.mean(), "%.2f" % psnrs.std(), evalrunner.scan_id
    )
    print(PSNR_MSG)

    psnrs = np.concatenate([psnrs, psnrs.mean()[None], psnrs.std()[None]])
    pd.DataFrame(psnrs).to_csv("{0}/psnr.csv".format(images_dir))

    ssims = np.array(ssims).astype(np.float64)
    SSIM_MSG = "ssim mean = {0} ; ssim std = {1}".format(
        "%.3f" % ssims.mean(), "%.3f" % ssims.std(), evalrunner.scan_id
    )
    print(SSIM_MSG)
    ssims = np.concatenate([ssims, ssims.mean()[None], ssims.std()[None]])
    pd.DataFrame(ssims).to_csv("{0}/ssim.csv".format(images_dir))

    lpipss = np.array(lpipss).astype(np.float64) * 100
    LPIPS_MSG = "lpips mean = {0} ; lpips std = {1}".format(
        "%.2f" % lpipss.mean(), "%.4f" % lpipss.std(), evalrunner.scan_id
    )
    print(LPIPS_MSG)
    lpipss = np.concatenate([lpipss, lpipss.mean()[None], lpipss.std()[None]])
    pd.DataFrame(lpipss).to_csv("{0}/lpips.csv".format(images_dir))

    MSG = PSNR_MSG + "\n" + SSIM_MSG + "\n" + LPIPS_MSG + "\n"
    logfile = f"{images_dir}/../{evalrunner.eval_method}.log"
    with open(logfile, "w") as f:
        f.write(MSG)
        
