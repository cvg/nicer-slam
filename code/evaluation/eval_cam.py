import argparse
import os
import numpy
import torch
import sys
import numpy as np
from easydict import EasyDict as edict
sys.path.append('code')
from utils.general import get_tensor_from_camera
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from copy import deepcopy

def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation
    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    """
    numpy.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = numpy.zeros((3, 3))
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity(3))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1) - rot * model.mean(1)

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error


def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib. 
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    """
    stamps.sort()
    interval = numpy.median([s-t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)


def evaluate_ate(first_list, second_list, Align=True, plot="", _args="", PLOT_TEXT=''):
    # parse command line
    parser = argparse.ArgumentParser(
        description='This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory.')
    # parser.add_argument('first_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    # parser.add_argument('second_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument(
        '--offset', help='time offset added to the timestamps of the second file (default: 0.0)', default=0.0)
    parser.add_argument(
        '--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument(
        '--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)', default=0.02)
    parser.add_argument(
        '--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations',
                        help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument(
        '--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument(
        '--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    args = parser.parse_args(_args)
    args.plot = plot
    # first_list = associate.read_file_list(args.first_file)
    # second_list = associate.read_file_list(args.second_file)

    matches = associate(first_list, second_list, float(
        args.offset), float(args.max_difference))
    if len(matches) < 2:
        raise ValueError(
            "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! \
            Did you choose the correct sequence?")
    # [[float(value) for value in first_list[a][0:3]] for a, b in matches]
    # for a, b in matches:
    #     print(floatfirst_list[a][0:3])
        # [float(value) for value in first_list[a][0:3]]
    first_xyz = numpy.matrix(
        [[float(value) for value in first_list[a][0:3]] for a, b in matches]).transpose()
    second_xyz = numpy.matrix([[float(value)*float(args.scale)
                              for value in second_list[b][0:3]] for a, b in matches]).transpose()

    rot, trans, trans_error = align(second_xyz, first_xyz)
    if not Align:
        rot = np.eye(3)
        trans = np.zeros_like(trans)

    second_xyz_aligned = rot * second_xyz + trans

    first_stamps = list(first_list.keys())
    first_stamps.sort()
    first_xyz_full = numpy.matrix(
        [[float(value) for value in first_list[b][0:3]] for b in first_stamps]).transpose()

    second_stamps = list(second_list.keys())
    second_stamps.sort()
    second_xyz_full = numpy.matrix([[float(value)*float(args.scale)
                                   for value in second_list[b][0:3]] for b in second_stamps]).transpose()
    second_xyz_full_aligned = rot * second_xyz_full + trans

    if args.verbose:
        print("compared_pose_pairs %d pairs" % (len(trans_error)))

        print("absolute_translational_error.rmse %f m" % numpy.sqrt(
            numpy.dot(trans_error, trans_error) / len(trans_error)))
        print("absolute_translational_error.mean %f m" %
              numpy.mean(trans_error))
        print("absolute_translational_error.median %f m" %
              numpy.median(trans_error))
        print("absolute_translational_error.std %f m" % numpy.std(trans_error))
        print("absolute_translational_error.min %f m" % numpy.min(trans_error))
        print("absolute_translational_error.max %f m" % numpy.max(trans_error))

    if args.save_associations:
        file = open(args.save_associations, "w")
        file.write("\n".join(["%f %f %f %f %f %f %f %f" % (a, x1, y1, z1, b, x2, y2, z2) for (
            a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A)]))
        file.close()

    if args.save:
        file = open(args.save, "w")
        file.write("\n".join(["%f " % stamp+" ".join(["%f" % d for d in line])
                   for stamp, line in zip(second_stamps, second_xyz_full_aligned.transpose().A)]))
        file.close()

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pylab as pylab
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ATE = numpy.sqrt(
            numpy.dot(trans_error, trans_error) / len(trans_error))
        ax.set_title(f'len:{len(trans_error)} {PLOT_TEXT    } ')
        plot_traj(ax, first_stamps, first_xyz_full.transpose().A,
                  '-', "black", "ground truth")
        plot_traj(ax, second_stamps, second_xyz_full_aligned.transpose(
        ).A, '-', "blue", "estimated")

        label = "difference"
        for (a, b), (x1, y1, z1), (x2, y2, z2) in zip(matches, first_xyz.transpose().A, second_xyz_aligned.transpose().A):
            # ax.plot([x1,x2],[y1,y2],'-',color="red",label=label)
            label = ""
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        # ax.set_xlim([-2, 1.2])
        # ax.set_ylim([-1, 1])
        plt.savefig(args.plot, dpi=90)

    return {
        "compared_pose_pairs": (len(trans_error)),
        "absolute_translational_error.rmse": numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error)),
        "absolute_translational_error.mean": numpy.mean(trans_error),
        "absolute_translational_error.median": numpy.median(trans_error),
        "absolute_translational_error.std": numpy.std(trans_error),
        "absolute_translational_error.min": numpy.min(trans_error),
        "absolute_translational_error.max": numpy.max(trans_error),
    }


def evaluate(poses_gt, poses_est, plot, Align=True, PLOT_TEXT=''):

    poses_gt = poses_gt.cpu().numpy()
    poses_est = poses_est.cpu().numpy()

    N = poses_gt.shape[0]
    poses_gt = dict([(i, poses_gt[i]) for i in range(N)])
    poses_est = dict([(i, poses_est[i]) for i in range(N)])

    results = evaluate_ate(poses_gt, poses_est, Align, plot, PLOT_TEXT=PLOT_TEXT)
    print(results)

class Pose():
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self,R=None,t=None):
        # construct a camera pose from the given R and/or t
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
        elif t is None:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1],device=R.device)
        else:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
        assert(pose.shape[-2:]==(3,4))
        return pose

    def invert(self,pose,use_inverse=False):
        # invert a camera pose
        R,t = pose[...,:3],pose[...,3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        pose_inv = self(R=R_inv,t=t_inv)
        return pose_inv

    def compose(self,pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new,pose)
        return pose_new

    def compose_pair(self,pose_a,pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a,t_a = pose_a[...,:3],pose_a[...,3:]
        R_b,t_b = pose_b[...,:3],pose_b[...,3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[...,0]
        pose_new = self(R=R_new,t=t_new)
        return pose_new
  
def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom
  
def cam2world(X,pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    re = X_hom@pose_inv.transpose(-1,-2)
    return re

def procrustes_analysis(X0,X1): # [N,3]
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t()).float()
    if R.det()<0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3


def prealign_cameras(pose, pose_GT):
    pose=Pose().invert(pose)
    pose_GT=Pose().invert(pose_GT)
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1,1,3, device='cpu')
    center_pred = cam2world(center,pose)[:,0] # [N,3]
    center_GT = cam2world(center,pose_GT)[:,0] # [N,3]
    try:
        sim3 = procrustes_analysis(center_GT,center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3,device='cpu'))
    # print(sim3)
    # align the camera poses
    center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
    R_aligned = pose[...,:3]@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
    pose=Pose()
    pose_aligned = pose(R=R_aligned,t=t_aligned)

    pose_aligned=Pose().invert(pose_aligned)
    return pose_aligned, sim3

def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

def evaluate_camera_alignment(pose_aligned,pose_GT):
    # measure errors in rotation and translation
    R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
    R_GT,t_GT = pose_GT.split([3,1],dim=-1)
    R_error = rotation_distance(R_aligned,R_GT)
    t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
    error = edict(R=R_error,t=t_error)
    return error
    

def convert_poses(c2w_list, N, scale, gt=True):
    poses = []
    for idx in range(0, N):
        c2w_list[idx][:3, 3] /= scale
        poses.append(get_tensor_from_camera(c2w_list[idx], Tquad=True))
    poses = torch.stack(poses)
    return poses 

if __name__ == '__main__':
    """
    This ATE evaluation code is modified upon the evaluation code in lie-torch.
    """

    parser = argparse.ArgumentParser(
        description='Arguments to eval the tracking ATE.'
    )
    parser.add_argument('--output', type=str, help='output folder')
    args = parser.parse_args()
    output = args.output
    scanid=int(output.split('/')[-2].split('_')[-1])
    if 'replica' in output:
        scalemat = np.load(f'Datasets/processed/Replica/scan{scanid}/cameras.npz')['scale_mat_0']
        scenes = ['', 'room0','room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4']
        scene=scenes[scanid]
        dataset='replica'
    elif '7scenes' in output:
        scalemat = np.load(f'Datasets/processed/7Scenes/scan{scanid}/cameras.npz')['scale_mat_0']
        scenes = ['', 'chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
        scene=scenes[scanid]
        dataset='7scenes'
    elif 'azure' in output:
        scalemat = np.load(f'Datasets/processed/Azure/scan{scanid}/cameras.npz')['scale_mat_0']
        scene=f'{scanid}'
        dataset='azure'
    elif 'demo' in output:
        scalemat = np.load(f'Datasets/processed/Demo/scan{scanid}/cameras.npz')['scale_mat_0']
        scene=f'{scanid}'
        dataset='demo'
    scale = 1/scalemat[0][0]
    ckptsdir = f'{output}/checkpoints/PoseParameters'
    if os.path.exists(ckptsdir): 
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'pth' in f]
        if len(ckpts) > 0:
            ckpt_path = ckpts[-1]
            print(ckpt_path)
            print('Get ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            estimate_c2w_list = ckpt['est_pose_all']
            gt_c2w_list = ckpt['gt_pose_all']
            gt_c2w_list=torch.stack(gt_c2w_list)

            estimate_c2w_list=list(estimate_c2w_list.values())
            estimate_c2w_list=torch.stack(estimate_c2w_list)

            gt_c2w_list=gt_c2w_list 
            estimate_c2w_list=estimate_c2w_list 

            gt_c2w_list[:, :3, 3]/=scale
            estimate_c2w_list[:, :3, 3]/=scale
            N = estimate_c2w_list.shape[0]
            gt_c2w_list=gt_c2w_list[:N][:, :3, :4]
            estimate_c2w_list=estimate_c2w_list[:N][:, :3, :4]
            select_idx=list(range(0, N, 1))
            

            gt_c2w_list=gt_c2w_list[select_idx]
            estimate_c2w_list=estimate_c2w_list[select_idx]

            # procrustes_analysis
            pose_aligned, _ = prealign_cameras(estimate_c2w_list, gt_c2w_list)
            bottoms = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
            [1, 4])).type(torch.float32).expand(pose_aligned.shape[0], 1, 4)
            aligned_estimate_c2w_list=torch.cat([pose_aligned, bottoms], dim=1)

            error = evaluate_camera_alignment(pose_aligned, gt_c2w_list)
            pose_aligned=convert_poses(pose_aligned, len(select_idx), 1)
            print("--------------------------")
            print("rot:   {:8.3f}".format(np.rad2deg(error.R.mean().cpu())))
            print("trans: {:10.5f}".format(error.t.mean()))
            print("--------------------------")
            PLOT_TEXT="rot: {:2.3f}deg".format(np.rad2deg(error.R.mean().cpu()))+" trans:{:2.2f}cm".format(error.t.mean()*100)

            # run evo
            if not os.path.exists(f'{output}/eval_cam'):
                original_estimate_c2w_list=deepcopy(estimate_c2w_list)
                original_estimate_c2w_list[:, :3, 3]*=scale
                N=original_estimate_c2w_list.shape[0]
                original_estimate_c2w_list = convert_poses(original_estimate_c2w_list, N, 1)
                # save to tum format
                tstamps = np.array(select_idx).reshape(-1, 1)
                tumformat=np.concatenate([tstamps, original_estimate_c2w_list], axis=-1)
                os.makedirs(f'{output}/eval_cam', exist_ok=True)
                seq_txt=f'{output}/eval_cam/traj.txt'
                np.savetxt(seq_txt, tumformat)
                gt_seq_txt=f'gt_trajs/gt_{dataset}_{scene}.txt'
                cmd=f'cd {output}/eval_cam; \
                    evo_ape tum ../../../../{gt_seq_txt} traj.txt --align --correct_scale -as -va  --plot --plot_mode xy --save_plot plot.png --save_results results.zip; \
                    unzip results.zip; rm results.zip;'
                os.system(cmd)
            
            gt_c2w_list = convert_poses(gt_c2w_list, len(select_idx), 1)
            evaluate(gt_c2w_list , pose_aligned,
                     plot=f'{output}/eval_cam_plot_{N:04d}.png', Align=True, PLOT_TEXT=PLOT_TEXT)
