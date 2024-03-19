import os
import time
import torch
import numpy as np
from glob import glob
import torch.nn.functional as F
from torchvision import transforms

def get_error_degrees(q1, q2):
    """
    Compute the rotation error (in degrees) between two quaternions.

    """
    # Normalize the quaternions
    q1_norm = F.normalize(q1, p=2, dim=-1)
    q2_norm = F.normalize(q2, p=2, dim=-1)

    # Compute the dot product of the two normalized quaternions
    dot_product = torch.dot(q1_norm, q2_norm)

    # Compute the rotation error as the arc-cosine of the dot product
    error = torch.acos(torch.abs(dot_product))

    # Convert the error to degrees
    error_degrees = error * (180 / torch.Tensor([3.14159265358979323846]).cuda())

    # print("Rotation error in degrees:", error_degrees.item())
    return error_degrees.item()


def index_to_1d(x, s):
    """
    Map 3D index to 1D index.

    """
    return x[:, 0] * s * s + x[:, 1] * s + x[:, 2]


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K


def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = -two_s * (qj * qj + qk * qk) + 1
    # rot_mat[:, 0, 0] = - rot_mat[:, 0, 0] + 1
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = -two_s * (qi**2 + qk**2) + 1
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = -two_s * (qi**2 + qj**2) + 1
    return rot_mat


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    bottom = (
        torch.from_numpy(np.array([0, 0, 0, 1.0]).reshape([1, 4]))
        .type(torch.float32)
        .to(RT.device)
        .unsqueeze(0)
        .repeat(RT.shape[0], 1, 1)
    )
    RT = torch.cat([RT, bottom], 1)
    if N == 1:
        RT = RT[0]
    return RT


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix

    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def uv2patch(uv, patchsize):
    """
    Given the center point of a patch and patch size, return the uv coordinates of the whole patch.

    """
    if patchsize == 1:
        patch_uv = uv.clone()
        patch_uv = patch_uv.reshape(-1, uv.shape[1], patchsize, patchsize, 2)
        return patch_uv
    half = patchsize // 2
    x = torch.tensor(range(-half, half + 1)).cuda()
    y = torch.tensor(range(-half, half + 1)).cuda()
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    gridxy = torch.stack([grid_x, grid_y], -1).unsqueeze(0).unsqueeze(0)
    uv = uv.unsqueeze(2).unsqueeze(2)
    patch_uv = uv + gridxy  # torch.Size([batch_size, N_pixels, patch_size, patch_size, 2])
    return patch_uv


def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def get_class(kls):
    parts = kls.split(".")
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def glob_imgs(path):
    imgs = []
    for ext in ["*.png", "*.jpg", "*.JPEG", "*.JPG"]:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def split_input(model_input, total_pixels, n_pixels=10000):
    """
    Split the input to fit Cuda memory for large resolution.
    Can decrease the value of n_pixels in case of cuda out of memory error.
    """
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data["uv"] = torch.index_select(model_input["uv"], 1, indx)
        if "object_mask" in data:
            data["object_mask"] = torch.index_select(model_input["object_mask"], 1, indx)
        if "depth" in data:
            data["depth"] = torch.index_select(model_input["depth"], 1, indx)
        if "gt_depth" in data:
            data["gt_depth"] = torch.index_select(model_input["gt_depth"], 1, indx)
        split.append(data)
    return split


def merge_output(res, total_pixels, batch_size):
    """Merge the split output."""

    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res], 1).reshape(
                batch_size * total_pixels
            )
        else:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res], 1
            ).reshape(batch_size * total_pixels, -1)

    return model_outputs


def concat_home_dir(path):
    return os.path.join(os.environ["HOME"], "data", path)


def get_time():
    torch.cuda.synchronize()
    return time.time()


trans_topil = transforms.ToPILImage()
