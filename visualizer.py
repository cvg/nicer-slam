import argparse
import os
from re import I
import time
import glob
import numpy as np
import torch
import cv2
from tqdm import tqdm
from copy import deepcopy
from code.utils.viz import SLAMFrontend
from easydict import EasyDict as edict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments to visualize the SLAM process.")
    parser.add_argument(
        "--output", type=str, help="output folder"
    )
    parser.add_argument(
        "--save_rendering", action="store_true", help="save rendering video to `vis.mp4` in output folder "
    )
    parser.add_argument(
        "--render_every_frame",
        action="store_true",
        help="render_every_frame, sync, good for comaprision",
    )
    parser.add_argument("--no_gt_traj", action="store_true", help="not visualize gt trajectory")
    args = parser.parse_args()
    output = args.output
    scanid = int(output.split("/")[-2].split("_")[-1])
    if "replica" in output or "investi" in output:
        dataset = "replica"
        scalemat = np.load(f'Datasets/processed/Replica/scan{scanid}/cameras.npz')['scale_mat_0']
        if scanid == 11:
            scanid = 1
        scenes = ["", "room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
        scene = scenes[scanid]
    elif "7scenes" in output:
        scalemat = np.load(f'Datasets/processed/7Scenes/scan{scanid}/cameras.npz')['scale_mat_0']
        scenes = ["", "chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
        scene = scenes[scanid]
        dataset = "7scenes"
    elif "azure" in output:
        scalemat = np.load(f'Datasets/processed/Azure/scan{scanid}/cameras.npz')['scale_mat_0']
        dataset = "azure"
    elif "demo" in output:
        scalemat = np.load(f'Datasets/processed/Demo/scan{scanid}/cameras.npz')['scale_mat_0']
        dataset = "demo"
    scalemat = torch.from_numpy(scalemat).float()
    ckptsdir = f"{output}/checkpoints/PoseParameters"
    if os.path.exists(ckptsdir):
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if "pth" in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print("Get ckpt :", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
            estimate_c2w_list = ckpt["est_pose_all"]
            gt_c2w_list = ckpt["gt_pose_all"]
            gt_c2w_list = torch.stack(gt_c2w_list)
            estimate_c2w_list = list(estimate_c2w_list.values())
            estimate_c2w_list = torch.stack(estimate_c2w_list)
            N = estimate_c2w_list.shape[0]

    sim3 = np.load(f"{output}/eval_cam/alignment_transformation_sim3.npy")
    estimate_c2w_list = torch.from_numpy(sim3).float() @ estimate_c2w_list
    gt_c2w_list = scalemat @ gt_c2w_list

    frontend = SLAMFrontend(
        output,
        init_pose=estimate_c2w_list[0],
        cam_scale=0.3,
        save_rendering=args.save_rendering,
        near=2,
        estimate_c2w_list=estimate_c2w_list,
        gt_c2w_list=gt_c2w_list,
        sim3=sim3,
        render_every_frame=args.render_every_frame,
    ).start()

    for i in tqdm(range(0, N, 1)):
        time.sleep(0.05)
        meshfile = f"{output}/vis/surface_{i:04d}.ply"
        if os.path.isfile(meshfile):
            frontend.update_mesh(meshfile)
        frontend.update_pose(0, estimate_c2w_list[i], gt=False)
        if not args.no_gt_traj:
            frontend.update_pose(1, gt_c2w_list[i], gt=True)

        if (i > 2) and (i % 2 == 0):
            frontend.update_cam_trajectory(i, gt=False)
            if not args.no_gt_traj:
                frontend.update_cam_trajectory(i, gt=True)

    if args.save_rendering:
        if args.render_every_frame:
            while len(glob.glob(f"{output}/tmp_rendering/*.jpg")) < N:
                time.sleep(1)

        time.sleep(1)
        os.system(
            f"/usr/bin/ffmpeg -f image2 -r 30 -pattern_type glob -i '{output}/tmp_rendering/*.jpg' -y {output}/vis.mp4"
        )
