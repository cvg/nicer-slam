import cv2
import torch
import trimesh
import numpy as np
import torchvision
from PIL import Image
from skimage import measure
import matplotlib.pyplot as plt


def plot(
    implicit_network,
    rendering_network,
    indices,
    plot_data,
    path,
    frame_idx,
    img_res,
    plot_nimgs,
    resolution,
    grid_boundary,
    level=0,
    inner_iter=0,
    save_mesh=False,
):
    if plot_data is not None:
        # plot images
        plot_images(
            plot_data["rgb_eval"],
            plot_data["rgb_gt"],
            path,
            frame_idx,
            plot_nimgs,
            img_res,
            indices,
            inner_iter=inner_iter,
        )

        # plot mono normal v.s. rendered normal maps
        plot_normal_maps(
            plot_data["normal_map"], plot_data["normal_gt"], path, frame_idx, plot_nimgs, img_res, indices, inner_iter
        )

        # plot mono depth v.s. rendered depth maps
        plot_depth_maps(
            plot_data["depth_map"], plot_data["depth_gt"], path, frame_idx, plot_nimgs, img_res, indices, inner_iter
        )

        # plot gt/sensor depth v.s. rendered depth maps
        plot_depth_maps(
            plot_data["depth_map"],
            plot_data["depth_real_gt"],
            path,
            frame_idx,
            plot_nimgs,
            img_res,
            indices,
            inner_iter,
            prefix="gt_",
        )

        # concat output images to single large image
        images = []
        for name in ["rendering", "depth", "normal", "gt_depth"]:
            images.append(
                cv2.imread("{0}/{1}_{2}_{3}_{4:04d}.png".format(path, name, frame_idx, indices[0], inner_iter))
            )
        # plot un images
        if "rgb_un_eval" in plot_data:
            plot_images(
                plot_data["rgb_eval"],
                plot_data["rgb_un_eval"],
                path,
                frame_idx,
                plot_nimgs,
                img_res,
                indices,
                exposure=True,
                inner_iter=inner_iter,
            )
            images.append(
                cv2.imread("{0}/{1}_{2}_{3}_{4:04d}.png".format(path, "exposure", frame_idx, indices[0], inner_iter))
            )
        images = np.concatenate(images, axis=0)
        cv2.imwrite("{0}/merge_{1}_{2}_{3:04d}.png".format(path, frame_idx, indices[0], inner_iter), images)

    if save_mesh:
        get_surface_trace(
            path=path,
            frame_idx=frame_idx,
            sdf=lambda x: implicit_network.get_sdf_vals(x)[:, 0],
            resolution=resolution,
            grid_boundary=grid_boundary,
            level=level,
            suffix="",
            color=True,
            implicit_network=implicit_network,
            rendering_network=rendering_network,
        )


def get_surface_trace(
    path,
    frame_idx,
    sdf,
    resolution=100,
    grid_boundary=[-2.0, 2.0],
    level=0,
    suffix="",
    color=False,
    implicit_network=None,
    rendering_network=None,
):
    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid["grid_points"]

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts.cuda()).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if not (np.min(z) > level or np.max(z) < level):

        z = z.astype(np.float32)

        z = z.reshape(grid["xyz"][1].shape[0], grid["xyz"][0].shape[0], grid["xyz"][2].shape[0]).transpose([1, 0, 2])

        verts, faces, vertex_normals, values = measure.marching_cubes(
            volume=z,
            level=level,
            spacing=(
                grid["xyz"][0][2] - grid["xyz"][0][1],
                grid["xyz"][0][2] - grid["xyz"][0][1],
                grid["xyz"][0][2] - grid["xyz"][0][1],
            ),
        )

        verts = verts + np.array([grid["xyz"][0][0], grid["xyz"][1][0], grid["xyz"][2][0]])

        if color and (not rendering_network.per_image_code):
            points_flat = torch.from_numpy(verts).float().cuda()
            sdf, feature_vectors, gradients = implicit_network.get_outputs(points_flat, stage="fine")
            dirs_flat = vertex_normals.reshape(-1, 3)
            dirs_flat *= -1
            dirs_flat = torch.from_numpy(dirs_flat.copy()).float().cuda()
            rgb_flat = rendering_network(
                points_flat, gradients, dirs_flat, feature_vectors, indices=None, color_stage="highfreq"
            )
            vertex_colors = rgb_flat.cpu().detach().numpy()
        else:
            vertex_colors = None
        meshexport = trimesh.Trimesh(verts, faces, vertex_normals=vertex_normals, vertex_colors=vertex_colors)
        meshexport.export("{0}/surface_{1:04d}{2}.ply".format(path, frame_idx, suffix), "ply")
    else:
        print("unable to get a surface, NO MESH!")


def get_grid_uniform(resolution, grid_boundary=[-2.0, 2.0]):
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points, "shortest_axis_length": 2.0, "xyz": [x, y, z], "shortest_axis_index": 0}


def plot_normal_maps(normal_maps, ground_true, path, frame_idx, plot_nrow, img_res, indices, inner_iter=0):
    ground_true = ground_true.cuda()

    residual = torch.abs(normal_maps - ground_true)
    normal_maps = torch.cat((normal_maps, ground_true, residual), dim=0)
    normal_maps_plot = lin2img(normal_maps, img_res)

    tensor = (
        torchvision.utils.make_grid(normal_maps_plot, scale_each=False, normalize=False, ncolumn=plot_nrow)
        .cpu()
        .detach()
        .numpy()
    )
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save("{0}/normal_{1}_{2}_{3:04d}.png".format(path, frame_idx, indices[0], inner_iter))


def plot_images(rgb_points, ground_true, path, frame_idx, plot_nrow, img_res, indices, exposure=False, inner_iter=0):
    ground_true = ground_true.cuda()
    residual = torch.abs(rgb_points - ground_true)
    output_vs_gt = torch.cat((rgb_points, ground_true, residual), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = (
        torchvision.utils.make_grid(output_vs_gt_plot, scale_each=False, normalize=False, ncolumn=plot_nrow)
        .cpu()
        .detach()
        .numpy()
    )

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    if exposure:
        img.save("{0}/exposure_{1}_{2}_{3:04d}.png".format(path, frame_idx, indices[0], inner_iter))
    else:
        img.save("{0}/rendering_{1}_{2}_{3:04d}.png".format(path, frame_idx, indices[0], inner_iter))


def plot_depth_maps(depth_maps, ground_true, path, frame_idx, plot_nrow, img_res, indices, inner_iter=0, prefix=""):
    ground_true = ground_true.cuda()
    residual = torch.abs(depth_maps[..., None] - ground_true)
    depth_maps = torch.cat((depth_maps[..., None], ground_true, residual), dim=0)
    depth_maps_plot = lin2img(depth_maps, img_res)
    depth_maps_plot = depth_maps_plot.expand(-1, 3, -1, -1)
    tensor = (
        torchvision.utils.make_grid(depth_maps_plot, scale_each=False, normalize=False, ncolumn=plot_nrow)
        .cpu()
        .detach()
        .numpy()
    )
    tensor = tensor.transpose(1, 2, 0)

    save_path = "{0}/{1}depth_{2}_{3}_{4:04d}.png".format(path, prefix, frame_idx, indices[0], inner_iter)
    plt.imsave(save_path, tensor[:, :, 0], cmap="plasma")


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
