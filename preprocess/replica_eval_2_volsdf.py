# this code is for converting test pose to the scaled coordinate system (to eval PSNR/SSIM)

import os
import cv2
import numpy as np
from tqdm import tqdm


scenes = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
ids = list(range(1, 9))
for id, scene in zip(ids, scenes):
    out_path = f"Datasets/processed/Replica_EVAL_EXT/scan{id}"
    os.makedirs(out_path, exist_ok=True)

    input_scalemat = f"Datasets/processed/Replica/scan{id}/cameras.npz"
    scale_mat = np.load(input_scalemat)["scale_mat_0"]

    data_root = f"Datasets/orig/Replica_eval_ext/{scene}"
    pose_file = os.path.join(data_root, "traj.txt")
    images_dir = data_root
    poses = np.loadtxt(pose_file)
    poses = poses.reshape(-1, 4, 4)
    c2w = poses
    num_image = c2w.shape[0]

    cameras = {}
    K = np.eye(4)
    K[0, 0] = 600.0
    K[1, 1] = 600.0
    K[0, 2] = 599.5
    K[1, 2] = 339.5

    for i in tqdm(range(num_image)):
        # copy rgb image file
        current_frame = os.path.join(images_dir, "frame%06d.jpg" % (i))
        target_image = os.path.join(out_path, "%06d_rgb.jpg" % (i))
        os.system("cp %s %s" % (current_frame, target_image))

        # # copy depth image file
        # current_frame = os.path.join(images_dir, 'depth%06d.png'%(i))
        # target_image = os.path.join(out_path, "%06d_gt_depth.png"%(i))
        # os.system("cp %s %s"%(current_frame, target_image))

        # save pose
        pose = c2w[i].copy()
        pose = K @ np.linalg.inv(pose)

        cameras["scale_mat_%d" % (i)] = scale_mat
        cameras["world_mat_%d" % (i)] = pose

    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)
