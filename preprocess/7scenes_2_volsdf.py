import os
import sys

os.environ["MKL_THREADING_LAYER"] = "GNU"
import cv2
import trimesh
import argparse
import numpy as np
from tqdm import tqdm

intrinsic_data = {
    "chess": {"fx": 535.30153598, "fy": 533.71239636, "cx": 316.85634818, "cy": 239.75744442},
    "fire": {"fx": 534.60449776, "fy": 539.02904297, "cx": 318.09034465, "cy": 248.36314533},
    "heads": {"fx": 533.48533767, "fy": 534.03326847, "cx": 315.07657519, "cy": 238.83690698},
    "office": {"fx": 534.924901, "fy": 549.31688003, "cx": 316.52655936, "cy": 256.39520434},
    "pumpkin": {"fx": 569.2724576, "fy": 544.82942106, "cx": 346.65669988, "cy": 221.8028837},
    "redkitchen": {"fx": 540.26264666, "fy": 545.1689031, "cx": 318.22221602, "cy": 246.72672228},
    "stairs": {"fx": 571.97464398, "fy": 570.18232961, "cx": 326.44024801, "cy": 238.53590499},
}


parser = argparse.ArgumentParser(description="Preprocess 7-Scenes dataset.")

parser.add_argument("--omnidata_path", dest="omnidata_path", help="path to omnidata model")
parser.set_defaults(omnidata_path="3rdparty/omnidata/omnidata_tools/torch/")

parser.add_argument("--pretrained_models", dest="pretrained_models", help="path to pretrained models")
parser.set_defaults(pretrained_models="3rdparty/omnidata/omnidata_tools/torch/pretrained_models/")

parser.add_argument("--gmflow_path", dest="gmflow_path", help="path to gmflow model")
parser.set_defaults(gmflow_path="3rdparty/gmflow")

parser.add_argument("--dataset_folder", dest="dataset_folder", help="path to dataset")
parser.set_defaults(dataset_folder="Datasets/orig/7Scenes")


args = parser.parse_args()

scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
ids = range(1, 8)
for id, scene in zip(ids, scenes):
    scene_data = intrinsic_data[scene]
    fx = scene_data["fx"]
    fy = scene_data["fy"]
    cx = scene_data["cx"]
    cy = scene_data["cy"]
    out_path = f"Datasets/processed/7Scenes/scan{id}"
    os.makedirs(out_path, exist_ok=True)

    seq = 1
    data_root = f"{args.dataset_folder}/{scene}/seq-{seq:02d}"

    c2ws = []
    if scene == "stairs":
        num_image = 500
    else:
        num_image = 1000
    for i in range(num_image):
        posefile = os.path.join(data_root, "frame-%06d.pose.txt" % (i))
        try:
            c2w = np.loadtxt(posefile)
        except FileNotFoundError:
            print("The file does not exist.")
            break
        c2ws.append(c2w)
    c2ws = np.stack(c2ws)
    cam_pos = c2ws[:, :3, 3]
    mesh_file = os.path.join(data_root, "../../meshes/%s.ply" % (scene))

    mesh = trimesh.load(mesh_file)

    min_vertices = mesh.vertices.min(axis=0)
    max_vertices = mesh.vertices.max(axis=0)
    min_vertices = np.minimum(cam_pos.min(axis=0), min_vertices)
    max_vertices = np.minimum(cam_pos.max(axis=0), max_vertices)

    center = (min_vertices + max_vertices) / 2.0
    scale = 2.0 / (np.max(max_vertices - min_vertices) * 1.5)

    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3] *= scale

    # now the scale_mat is from the original to the normalized
    mesh.vertices = mesh.vertices @ scale_mat[:3, :3].T + scale_mat[:3, 3]
    mesh.export(out_path + "/../%s_mesh_%02d.ply" % (scene, id))

    # now the scale_mat is from the normalized to the original
    scale_mat = np.linalg.inv(scale_mat)

    # the scale_mat will be multiplied into the pose in the main code
    out_index = 0
    cameras = {}
    K = np.eye(4)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
    for i in tqdm(range(num_image)):
        # copy rgb image file
        current_frame = os.path.join(data_root, "frame-%06d.color.png" % (i))
        target_image = os.path.join(out_path, "%06d_rgb.png" % (i))
        os.system("cp %s %s" % (current_frame, target_image))

        # copy depth image file
        current_frame = os.path.join(data_root, "frame-%06d.depth.png" % (i))
        depth = cv2.imread(current_frame, -1)
        # 7scenes use 65535 to represent invalid pixel, we use 0
        depth[depth == 65535] = 0
        target_image = os.path.join(out_path, "%06d_gt_depth.png" % (i))
        cv2.imwrite(target_image, depth)
        os.system("cp %s %s" % (current_frame, target_image))

        # save pose
        pose = c2ws[i]
        pose = K @ np.linalg.inv(pose)

        cameras["scale_mat_%d" % (i)] = scale_mat
        cameras["world_mat_%d" % (i)] = pose

    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)

    # extract monocular cues
    python_executable_path = sys.executable
    python_path = python_executable_path.replace("nicer-slam", "omnidata")
    os.system(
        f"{python_path} preprocess/extract_monocular_cues.py --task depth --img_path {out_path} --output_path {out_path} --omnidata_path {args.omnidata_path} --pretrained_models {args.pretrained_models}"
    )
    os.system(
        f"{python_path} preprocess/extract_monocular_cues.py --task normal --img_path {out_path} --output_path {out_path} --omnidata_path {args.omnidata_path} --pretrained_models {args.pretrained_models}"
    )
    # extract flow
    python_path = python_executable_path.replace("nicer-slam", "gmflow")
    os.system(
        f"{python_path} preprocess/extract_flows.py --inference_dir {out_path} --output_path {out_path}_pair --gmflow_path {args.gmflow_path} --fwd_bwd_consistency_check --pred_bidir_flow --resume {args.gmflow_path}/pretrained/gmflow_sintel-0c07dcb3.pth"
    )
