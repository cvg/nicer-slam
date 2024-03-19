import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
import sys
import trimesh
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Preprocess Replica dataset.")

parser.add_argument("--omnidata_path", dest="omnidata_path", help="path to omnidata model")
parser.set_defaults(omnidata_path="3rdparty/omnidata/omnidata_tools/torch/")

parser.add_argument("--pretrained_models", dest="pretrained_models", help="path to pretrained models")
parser.set_defaults(pretrained_models="3rdparty/omnidata/omnidata_tools/torch/pretrained_models/")

parser.add_argument("--gmflow_path", dest="gmflow_path", help="path to gmflow model")
parser.set_defaults(gmflow_path="3rdparty/gmflow")

parser.add_argument("--dataset_folder", dest="dataset_folder", help="path to dataset")
parser.set_defaults(dataset_folder="Datasets/orig/Replica")


args = parser.parse_args()

scenes = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
ids = range(1, 9)
for id, scene in zip(ids, scenes):
    out_path = f"Datasets/processed/Replica/scan{id}"
    os.makedirs(out_path, exist_ok=True)

    data_root = f"{args.dataset_folder}/{scene}"
    pose_file = os.path.join(data_root, "traj.txt")
    images_dir = os.path.join(data_root, "results")

    mesh_file = os.path.join(data_root, "../%s_mesh.ply" % (scene))

    mesh = trimesh.load(mesh_file)

    min_vertices = mesh.vertices.min(axis=0)
    max_vertices = mesh.vertices.max(axis=0)

    center = (min_vertices + max_vertices) / 2.0
    if id in [1, 2, 3]:
        scale = 2.0 / (np.max(max_vertices - min_vertices) * 1.5)
    else:
        scale = 2.0 / (np.max(max_vertices - min_vertices) * 2)
    poses = np.loadtxt(pose_file)
    poses = poses.reshape(-1, 4, 4)

    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3] *= scale

    # now the scale_mat is from the original to the normalized
    mesh.vertices = mesh.vertices @ scale_mat[:3, :3].T + scale_mat[:3, 3]
    mesh.export(out_path + "/../%s_mesh_%02d.ply" % (scene, id))

    # now the scale_mat is from the normalized to the original
    scale_mat = np.linalg.inv(scale_mat)

    # the scale_mat will be multiplied into the pose in the main code
    c2w = poses
    num_image = c2w.shape[0]

    out_index = 0
    cameras = {}
    K = np.eye(4)
    K[0, 0] = 600.0
    K[1, 1] = 600.0
    K[0, 2] = 599.5
    K[1, 2] = 339.5

    for i in tqdm(range(num_image)):
        # copy rgb image file
        current_frame = os.path.join(images_dir, 'frame%06d.jpg'%(i))
        target_image = os.path.join(out_path, "%06d_rgb.png"%(i))
        os.system("cp %s %s"%(current_frame, target_image))

        # copy depth image file
        current_frame = os.path.join(images_dir, 'depth%06d.png'%(i))
        target_image = os.path.join(out_path, "%06d_gt_depth.png"%(i))
        os.system("cp %s %s"%(current_frame, target_image))

        # save pose
        pose = c2w[i].copy()
        pose = K @ np.linalg.inv(pose)

        cameras["scale_mat_%d"%(i)] = scale_mat
        cameras["world_mat_%d"%(i)] = pose

    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)

    # extract monocular cues
    python_executable_path = sys.executable
    python_path = python_executable_path.replace('nicer-slam', 'omnidata')
    os.system(f"{python_path} preprocess/extract_monocular_cues.py --task depth --img_path {out_path} --output_path {out_path} --omnidata_path {args.omnidata_path} --pretrained_models {args.pretrained_models}")
    os.system(f"{python_path} preprocess/extract_monocular_cues.py --task normal --img_path {out_path} --output_path {out_path} --omnidata_path {args.omnidata_path} --pretrained_models {args.pretrained_models}")
    # extract flow
    python_path = python_executable_path.replace('nicer-slam', 'gmflow')
    os.system(f"{python_path} preprocess/extract_flows.py --inference_dir {out_path} --output_path {out_path}_pair --gmflow_path {args.gmflow_path} --fwd_bwd_consistency_check --pred_bidir_flow --resume {args.gmflow_path}/pretrained/gmflow_sintel-0c07dcb3.pth")
