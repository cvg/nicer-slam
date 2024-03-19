import os
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

dataset_root = "Datasets/orig/7Scenes"
scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

for scene in scenes:
    seq = 1
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0, sdf_trunc=0.04, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    if scene == "stairs":
        n_imgs = 500
    else:
        n_imgs = 1000
    for i in tqdm(range(0, n_imgs, 1)):
        data_root = f"{dataset_root}/{scene}/seq-{seq:02d}"
        posefile = os.path.join(data_root, "frame-%06d.pose.txt" % (i))
        c2w = np.loadtxt(posefile)

        rgbfile = os.path.join(data_root, "frame-%06d.color.png" % (i))
        color = cv2.imread(rgbfile)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depthfile = os.path.join(data_root, "frame-%06d.depth.png" % (i))
        depth = cv2.imread(depthfile, -1)
        depth[depth == 65535] = 0.0
        depth = depth / 1000
        color = o3d.geometry.Image(color.astype(np.int8))
        depth = o3d.geometry.Image(depth.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1, depth_trunc=100.0, convert_rgb_to_intensity=False
        )
        # not accurate but good enough for TSDF fusion
        fx, fy, cx, cy = 585, 585, 320, 240
        camera = o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)
        w2c = np.linalg.inv(c2w)
        volume.integrate(rgbd, camera, w2c)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    os.makedirs(f"{dataset_root}/meshes", exist_ok=True)
    o3d.io.write_triangle_mesh(f"{dataset_root}/meshes/{scene}.ply", mesh)
