import os
import sys
import glob
import trimesh
import argparse
import numpy as np

import open3d as o3d
from scipy.spatial import cKDTree as KDTree


def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances, indices


def eval_pointcloud(
    pointcloud, pointcloud_tgt, normals=None, normals_tgt=None, thresholds=np.linspace(1.0 / 1000, 1, 1000)
):
    """Evaluates a point cloud.

    Args:
        pointcloud (numpy array): predicted point cloud
        pointcloud_tgt (numpy array): target point cloud
        normals (numpy array): predicted normals
        normals_tgt (numpy array): target normals
        thresholds (numpy array): threshold values for the F-score calculation
    """
    # Return maximum losses if pointcloud is empty
    if pointcloud.shape[0] == 0:
        logger.warn("Empty pointcloud / mesh detected!")
        out_dict = EMPTY_PCL_DICT.copy()
        if normals is not None and normals_tgt is not None:
            out_dict.update(EMPTY_PCL_DICT_NORMALS)
        return out_dict

    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, completeness_normals = distance_p2p(pointcloud_tgt, normals_tgt, pointcloud, normals)
    # print('completeness_normals', completeness_normals)
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(pointcloud, normals, pointcloud_tgt, normals_tgt)
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
    chamferL1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]

    out_dict = {
        "completeness": completeness,
        "accuracy": accuracy,
        "normals completeness": completeness_normals,
        "normals accuracy": accuracy_normals,
        "normals": normals_correctness,
        "completeness2": completeness2,
        "accuracy2": accuracy2,
        "chamfer-L2": chamferL2,
        "chamfer-L1": chamferL1,
        "f-score": F[9],  # threshold = 1.0%
        "f-score-15": F[14],  # threshold = 1.5%
        "f-score-20": F[19],  # threshold = 2.0%
    }

    return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def distance_p2m(points, mesh):
    """Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    """
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def get_threshold_percentage(dist, thresholds):
    """Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).mean() for t in thresholds]
    return in_threshold


def calc_normal_consistency(rec_meshfile, gt_meshfile, align=True, scale=1):
    """
    3D reconstruction metric.

    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    mesh_rec.vertices /= scale
    mesh_gt.vertices /= scale

    if align:
        transformation = get_align_transformation(mesh_rec, mesh_gt)
        mesh_rec = mesh_rec.apply_transform(transformation)
    num_points = 200000
    rec_pointcloud, idx = mesh_rec.sample(num_points, return_index=True)
    rec_pointcloud = rec_pointcloud.astype(np.float32)
    rec_normals = mesh_rec.face_normals[idx]
    gt_pointcloud, idx = mesh_gt.sample(num_points, return_index=True)
    gt_pointcloud = gt_pointcloud.astype(np.float32)
    gt_normals = mesh_gt.face_normals[idx]

    out_dict = eval_pointcloud(rec_pointcloud, gt_pointcloud, rec_normals, gt_normals)
    print("Normal Consistency", f"{out_dict['normals']*100:.4f} %")


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float))
    return comp_ratio


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def get_align_transformation(mesh_rec, mesh_gt):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    # o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    # o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(mesh_rec.vertices))
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(mesh_gt.vertices))
    trans_init = np.eye(4)
    threshold = 0.1
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc, o3d_gt_pc, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    transformation = reg_p2p.transformation
    return transformation


def calc_3d_metric(rec_meshfile, gt_meshfile, align=True, scale=1):
    """
    3D reconstruction metric.

    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False)

    mesh_rec.vertices /= scale
    mesh_gt.vertices /= scale

    if align:
        transformation = get_align_transformation(mesh_rec, mesh_gt)
        mesh_rec = mesh_rec.apply_transform(transformation)

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices)
    accuracy_rec *= 100  # convert to cm
    completion_rec *= 100  # convert to cm
    completion_ratio_rec *= 100  # convert to %
    print("accuracy: ", accuracy_rec, "cm")
    print("completion: ", completion_rec, "cm")
    print("completion ratio: ", completion_ratio_rec, "%")


if __name__ == "__main__":
    """
    This 3D evaluation code is modified upon the evaluation code in NICE_SLAM/ConvONet.
    """

    parser = argparse.ArgumentParser(description="Arguments to eval the 3D reconstruction.")
    parser.add_argument("--output", type=str, help="output folder")

    args = parser.parse_args()
    output = args.output
    scanid = int(output.split("/")[-2].split("_")[-1])

    if "replica" in output:
        dataset = "replica"
        scenes = ["", "room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
        scene = scenes[scanid]
    elif "7scenes" in output:
        dataset = "7scenes"
    elif "azure" in output:
        dataset = "azure"

    rec_mesh_file = sorted(glob.glob(f"{output}/vis/surface_*0.ply"))[-1]
    sim3 = np.load(f"{output}/eval_cam/alignment_transformation_sim3.npy")
    rec_mesh = o3d.io.read_triangle_mesh(rec_mesh_file)
    aligned_rec_mesh = rec_mesh.transform(sim3)
    aligned_rec_mesh_file = f"{output}/mesh_aligned_first.ply"
    o3d.io.write_triangle_mesh(aligned_rec_mesh_file, aligned_rec_mesh)
    # for dataset other than replica, only alignment is done, since no GT mesh is available
    if dataset == "replica":
        gt_mesh = f"Datasets/orig/Replica/{scene}_mesh.ply"
        second_aligned_rec_mesh_file = f"{output}/mesh_aligned.ply"
        # sometimes there might be a large connected component outside the mesh, it can be easily removed by using 'Select Connected Components in a region' tool in Meshlab
        input(
            "After mannually align using CloudCompare, input:mesh_aligned_first.ply, reference: gt_mesh, output:mesh_aligned.ply, press any key to continue"
        )
        # on some computer it is also possible to run the CloudCompare command line
        # cmd=f'/snap/bin/cloudcompare.CloudCompare -SILENT -O "{aligned_rec_mesh_file}" -O "{gt_mesh}"  -M_EXPORT_FMT ply -ICP -ADJUST_SCALE -AUTO_SAVE ON -SAVE_MESHES FILE "{second_aligned_rec_mesh_file} /dev/shm/tmp.ply"'
        # os.system(cmd)
        calc_3d_metric(second_aligned_rec_mesh_file, gt_mesh, scale=1)
        calc_normal_consistency(second_aligned_rec_mesh_file, gt_mesh, scale=1)
