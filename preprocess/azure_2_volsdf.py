import numpy as np
import sys
import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
from tqdm import tqdm
import trimesh
import argparse
import glob

from colmap_utils.pose_utils import load_colmap_data


parser = argparse.ArgumentParser(description="Preprocess Azure dataset.")

parser.add_argument("--omnidata_path", dest="omnidata_path", help="path to omnidata model")
parser.set_defaults(omnidata_path="3rdparty/omnidata/omnidata_tools/torch/")

parser.add_argument("--pretrained_models", dest="pretrained_models", help="path to pretrained models")
parser.set_defaults(pretrained_models="3rdparty/omnidata/omnidata_tools/torch/pretrained_models/")

parser.add_argument("--gmflow_path", dest="gmflow_path", help="path to gmflow model")
parser.set_defaults(gmflow_path="3rdparty/gmflow")

parser.add_argument("--dataset_folder", dest="dataset_folder", help="path to dataset")
parser.set_defaults(dataset_folder="Datasets/orig/Azure")


args = parser.parse_args()


def read_colmap_pose(basedir):
    files_needed = ["{}.bin".format(f) for f in ["cameras", "images", "points3D"]]
    if os.path.exists(os.path.join(basedir, "sparse/0")):
        files_had = os.listdir(os.path.join(basedir, "sparse/0"))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print("Need to run COLMAP")
        exit(0)
    else:
        print("Don't need to run COLMAP")

    print("Post-colmap")

    poses, intrinsics, pts3d, perm = load_colmap_data(basedir, return_intrinsics=True)
    poses = poses.transpose(2, 0, 1)
    poses = poses[:, :, :4]
    return poses, intrinsics


scenes = ["1", "2", "3", "4", "5", "6"]
ids = range(1, 7)

resize = 1  # each image is resized to H//resize, W//resize
for id, scene in zip(ids, scenes):
    out_path = f"Datasets/processed/Azure/scan{id}"
    os.makedirs(out_path, exist_ok=True)

    data_root = f"{args.dataset_folder}/{scene}"

    poses, intrinsics = read_colmap_pose(data_root)
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    fx, fy, cx, cy = fx / resize, fy / resize, cx / resize, cy / resize
    images_dir = os.path.join(data_root, f"color")
    if id in [1, 2, 3, 6]:
        mesh_file = os.path.join(data_root, "dense/meshed-poisson.ply")
    else:
        mesh_file = os.path.join(data_root, "dense/meshed-delaunay.ply")

    mesh = trimesh.load(mesh_file)

    min_vertices = mesh.vertices.min(axis=0)
    max_vertices = mesh.vertices.max(axis=0)
    min_pose = np.min(poses[:, :3, 3], axis=0)
    max_pose = np.max(poses[:, :3, 3], axis=0)
    min_vertices = np.minimum(min_vertices, min_pose)
    max_vertices = np.maximum(max_vertices, max_pose)

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
    c2ws = poses
    num_image = c2ws.shape[0]

    cameras = {}
    K = np.eye(4)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    rgbfiles = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    print(id, num_image, len(rgbfiles))
    for i in tqdm(range(num_image)):
        # copy rgb image file
        current_frame = rgbfiles[i]
        target_image = os.path.join(out_path, "%06d_rgb.png" % (i))
        os.system("cp %s %s" % (current_frame, target_image))

        # copy depth image file
        # current_frame = os.path.join(images_dir, '..', 'depth%06d.png'%(i))
        # target_image = os.path.join(out_path, "%06d_gt_depth.png"%(i))
        # os.system("cp %s %s"%(current_frame, target_image))

        # save pose
        pose = np.eye(4)
        pose[:3, :] = c2ws[i].copy()
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
