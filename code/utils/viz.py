import os
from multiprocessing import Process, Queue
from queue import Empty

import numpy as np
import open3d as o3d
import torch


def remove_scale_from_camera_pose(pose_matrix):
    # pose_matrix is a 4x4 numpy array

    # For each of the first three columns, normalize to remove scale
    for i in range(3):
        column = pose_matrix[:, i]
        scale_factor = np.linalg.norm(column[:3])  # Compute the norm of the column, excluding the bottom element
        if scale_factor > 0:  # Avoid division by zero
            pose_matrix[:, i] /= scale_factor  # Normalize column to remove scale

    # The last column (translation) and the last row are not modified, as they do not contribute to scale
    return pose_matrix


def normalize(x):
    return x / np.linalg.norm(x)


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


def create_camera_actor(i, is_gt=False, scale=0.005):
    cam_points = scale * np.array(
        [
            [0, 0, 0],
            [-1, -1, 1.5],
            [1, -1, 1.5],
            [1, 1, 1.5],
            [-1, 1, 1.5],
            [-0.5, 1, 1.5],
            [0.5, 1, 1.5],
            [0, 1.2, 1.5],
        ]
    )

    cam_lines = np.array(
        [[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
    )
    points = []
    for cam_line in cam_lines:
        begin_points, end_points = cam_points[cam_line[0]], cam_points[cam_line[1]]
        t_vals = np.linspace(0.0, 1.0, 100)
        begin_points, end_points
        point = begin_points[None, :] * (1.0 - t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
        points.append(point)
    points = np.concatenate(points)
    color = (0.0, 0.0, 0.0) if is_gt else (1.0, 0.0, 0.0)
    camera_actor = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(color)

    return camera_actor


def draw_trajectory(
    queue, output, init_pose, cam_scale, save_rendering, near, estimate_c2w_list, gt_c2w_list, sim3, render_every_frame
):

    draw_trajectory.queue = queue
    draw_trajectory.cameras = {}
    draw_trajectory.points = {}
    draw_trajectory.ix = 0
    draw_trajectory.warmup = 0
    draw_trajectory.mesh = None
    draw_trajectory.frame_idx = 0
    draw_trajectory.traj_actor = None
    draw_trajectory.traj_actor_gt = None
    draw_trajectory.pose = False

    if save_rendering:
        os.system(f"rm -rf {output}/tmp_rendering")

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        draw_trajectory.pose = False
        while True:
            try:
                data = draw_trajectory.queue.get_nowait()
                if data[0] == "pose":
                    i, pose, is_gt = data[1:]
                    pose = remove_scale_from_camera_pose(pose)
                    draw_trajectory.pose = not is_gt
                    if is_gt:
                        i += 100000

                    if i in draw_trajectory.cameras:
                        cam_actor, pose_prev = draw_trajectory.cameras[i]
                        pose_change = pose @ np.linalg.inv(pose_prev)

                        cam_actor.transform(pose_change)
                        vis.update_geometry(cam_actor)

                        if i in draw_trajectory.points:
                            pc = draw_trajectory.points[i]
                            pc.transform(pose_change)
                            vis.update_geometry(pc)

                    else:
                        cam_actor = create_camera_actor(i, is_gt, cam_scale)
                        cam_actor.transform(pose)
                        vis.add_geometry(cam_actor)

                    draw_trajectory.cameras[i] = (cam_actor, pose)
                    if render_every_frame:
                        break
                elif data[0] == "mesh":
                    meshfile = data[1]
                    if draw_trajectory.mesh is not None:
                        vis.remove_geometry(draw_trajectory.mesh)
                    
                    # To only keep the largest connected components
                    # import trimesh
                    # mesh = trimesh.load(meshfile)
                    # components = mesh.split(only_watertight=False)
                    # areas = np.array([c.area for c in components], dtype=np.float32)
                    # mesh_clean = components[areas.argmax()]
                    # mesh_clean.export('tmp.ply')
                    # mesh = o3d.io.read_triangle_mesh('tmp.ply')
                    
                    mesh = o3d.io.read_triangle_mesh(meshfile)
                    mesh = mesh.transform(sim3)
                    draw_trajectory.mesh = mesh
                    draw_trajectory.mesh.compute_vertex_normals()
                    vis.add_geometry(draw_trajectory.mesh)

                # use pointcloud to represent trajectory, good for dense input
                elif data[0] == "traj":
                    i, is_gt = data[1:]

                    color = (0.0, 0.0, 0.0) if is_gt else (1.0, 0.0, 0.0)
                    traj_actor = o3d.geometry.PointCloud(
                        points=o3d.utility.Vector3dVector(
                            gt_c2w_list[1:i, :3, 3] if is_gt else estimate_c2w_list[1:i, :3, 3]
                        )
                    )
                    traj_actor.paint_uniform_color(color)

                    if is_gt:
                        if draw_trajectory.traj_actor_gt is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor_gt)
                            tmp = draw_trajectory.traj_actor_gt
                            del tmp
                        draw_trajectory.traj_actor_gt = traj_actor
                        vis.add_geometry(draw_trajectory.traj_actor_gt)
                    else:
                        if draw_trajectory.traj_actor is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor)
                            tmp = draw_trajectory.traj_actor
                            del tmp
                        draw_trajectory.traj_actor = traj_actor
                        vis.add_geometry(draw_trajectory.traj_actor)

                elif data[0] == "reset":
                    draw_trajectory.warmup = -1

                    for i in draw_trajectory.points:
                        vis.remove_geometry(draw_trajectory.points[i])

                    for i in draw_trajectory.cameras:
                        vis.remove_geometry(draw_trajectory.cameras[i][0])

                    draw_trajectory.cameras = {}
                    draw_trajectory.points = {}

            except Empty:
                break

        # hack to allow interacting with vizualization during inference
        if len(draw_trajectory.cameras) >= draw_trajectory.warmup:
            cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

        vis.poll_events()
        vis.update_renderer()
        if save_rendering:
            if render_every_frame:
                # print('render_every_frame', render_every_frame)
                # print('draw_trajectory.pose', draw_trajectory.pose)
                if draw_trajectory.pose:
                    # i, points, colors = data[1:]
                    draw_trajectory.frame_idx += 1
                    os.makedirs(f"{output}/tmp_rendering", exist_ok=True)
                    vis.capture_screen_image(f"{output}/tmp_rendering/{draw_trajectory.frame_idx:06d}.jpg")
            else:
                # save the renderings, useful when making a video
                draw_trajectory.frame_idx += 1
                os.makedirs(f"{output}/tmp_rendering", exist_ok=True)
                vis.capture_screen_image(f"{output}/tmp_rendering/{draw_trajectory.frame_idx:06d}.jpg")

    vis = o3d.visualization.Visualizer()
    vis.register_animation_callback(animation_callback)
    vis.create_window(window_name=output, height=990, width=1760)
    vis.get_render_option().point_size = 4
    vis.get_render_option().mesh_show_back_face = False
    ctr = vis.get_view_control()
    ctr.set_constant_z_near(near)
    ctr.set_constant_z_far(1000)

    # set the viewer's pose in the back of the first frame's pose
    param = ctr.convert_to_pinhole_camera_parameters()
    init_pose[:3, 3] -= 0.2 * normalize(init_pose[:3, 2])
    init_pose = np.linalg.inv(init_pose)

    param.extrinsic = init_pose
    ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()
    vis.destroy_window()


class SLAMFrontend:
    def __init__(
        self,
        output,
        init_pose,
        cam_scale=1,
        save_rendering=False,
        near=0,
        estimate_c2w_list=None,
        gt_c2w_list=None,
        sim3=None,
        render_every_frame=False,
    ):
        self.queue = Queue()
        self.p = Process(
            target=draw_trajectory,
            args=(
                self.queue,
                output,
                init_pose,
                cam_scale,
                save_rendering,
                near,
                estimate_c2w_list,
                gt_c2w_list,
                sim3,
                render_every_frame,
            ),
        )

    def update_pose(self, index, pose, gt=False):
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()

        pose[:3, 1] *= -1
        self.queue.put_nowait(("pose", index, pose, gt))

    def update_mesh(self, path):
        self.queue.put_nowait(("mesh", path))

    def update_cam_trajectory(self, c2w_list, gt):
        self.queue.put_nowait(("traj", c2w_list, gt))

    def reset(self):
        self.queue.put_nowait(("reset",))

    def start(self):
        self.p.start()
        return self

    def join(self):
        self.p.join()
