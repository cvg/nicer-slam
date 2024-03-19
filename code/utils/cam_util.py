import torch
from easydict import EasyDict as edict


class Pose:
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self, R=None, t=None):
        # construct a camera pose from the given R and/or t
        assert R is not None or t is not None
        if R is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1)  # [...,3,4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False):
        # invert a camera pose
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    re = X_hom @ pose_inv.transpose(-1, -2)
    return re


def procrustes_analysis(X0, X1):  # [N,3]
    # translation
    t0 = X0.mean(dim=0, keepdim=True)
    t1 = X1.mean(dim=0, keepdim=True)
    X0c = X0 - t0
    X1c = X1 - t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c / s0
    X1cs = X1c / s1
    # rotation (use double for SVD, float loses precision)
    U, S, V = (X0cs.t() @ X1cs).double().svd(some=True)
    R = (U @ V.t()).float()
    if R.det() < 0:
        R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0], t1=t1[0], s0=s0, s1=s1, R=R)
    return sim3


def prealign_cameras_apply_another(pose, pose_GT, apply_pose):
    pose = Pose().invert(pose)
    pose_GT = Pose().invert(pose_GT)
    apply_pose = Pose().invert(apply_pose)
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1, 1, 3, device="cuda:0")
    center_pred = cam2world(center, pose)[:, 0]  # [N,3]
    center_GT = cam2world(center, pose_GT)[:, 0]  # [N,3]
    center_apply_pose = cam2world(center, apply_pose)[:, 0]  # [N,3]
    try:
        sim3 = procrustes_analysis(center_GT, center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device="cuda:0"))
    # align the camera poses
    center_aligned = (center_apply_pose - sim3.t1) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
    R_aligned = apply_pose[..., :3] @ sim3.R.t()
    t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
    pose = Pose()
    pose_aligned = pose(R=R_aligned, t=t_aligned)
    pose_aligned = Pose().invert(pose_aligned)
    return pose_aligned, sim3
