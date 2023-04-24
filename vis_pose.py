import sys
import os
import argparse
from pathlib import Path
from torch.autograd import Variable

sys.path.append(os.path.join(sys.path[0], '../..'))

import open3d as o3d
import torch
import numpy as np

from utils.lie_group_helper import convert3x4_4x4
from data_loader import llff_data
from models import Net,Block
import torchvision
from PIL import Image
from mmflow.apis import init_model, inference_model
from mmflow.datasets import visualize_flow, write_flow
from utils.align_traj import align_ate_c2b_use_a2b, pts_dist_max
from utils.comp_ate import compute_ate
from utils.vis_cam_traj import draw_camera_frustum_geometry

def vis(scene_file, model_file):
    my_devices = torch.device('cpu')
    basedir = "./nerf_llff_data/fern"
    datas = llff_data(basedir, factor=8)
    datas.load_data()



    '''Load scene meta'''
    H, W ,colmap_focal = datas.hwf[0], datas.hwf[1], datas.hwf[2]

    print('Intrinsic: H: {0:4f}, W: {1:4f}, COLMAP focal {2:.2f}.'.format(H, W, colmap_focal))

    '''Model Loading'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(Block)
    net.load_state_dict(torch.load(model_file))
    net.eval()
    net.to(device)



    '''Get all poses in (N, 4, 4)'''
    c2ws_cmp=datas.poses
    img0_pos=c2ws_cmp[0]
    c2ws_cmp=convert3x4_4x4(c2ws_cmp)
    c2ws_cmp=[torch.from_numpy(i).float() for i in c2ws_cmp]
    c2ws_cmp = torch.stack(c2ws_cmp)
    ts_colmap = c2ws_cmp[:, :3, 3]  # (N, 3)

    c2ws_cmp[:, :3, 3] /= pts_dist_max(ts_colmap)
    c2ws_cmp[:, :3, 3] *= 2.0

    c2ws_est=[]
    c2ws_est.append(img0_pos)

    image_names = [os.path.join(basedir, 'images_8', f) for f in sorted(os.listdir(os.path.join(basedir, 'images_8'))) \
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    config_file = './flownet2css_8x1_sfine_flyingthings3d_subset_384x768.py'
    checkpoint_file = 'flownet2css_8x1_sfine_flyingthings3d_subset_384x768.pth'
    pwc_model = init_model(config_file, checkpoint_file, device='cuda:0')
    pwc_model.eval()
    for i in range(len(image_names)-1):
        optical_flow = inference_model(pwc_model, image_names[i], image_names[i+1])
        optical_flow = torch.from_numpy(np.asarray([optical_flow])).float()
        optical_flow = optical_flow.permute(0, 3, 1, 2)
        optical_flow = Variable(optical_flow.to(device))
        pred_translation, pred_rotation = net(optical_flow)
        pose_1=c2ws_est[-1]
        pose_1=datas.poses[i]
        pose_2=pose_1.copy()
        pose_2=pose_2.T

        pose_2[3:,:]=np.multiply(pred_translation.cpu().detach().numpy(), 0.30273836851119995)+pose_2[-1]

        pose_2[:3, :]=np.dot(pose_2[:3, :],pred_rotation.cpu().detach().numpy())
        pose_2=pose_2.T
        c2ws_est.append(pose_2)
    c2ws_est=[torch.from_numpy(i).float() for i in c2ws_est ]
    c2ws_est = torch.stack(c2ws_est)  # (N, 4, 4)
    c2ws_est = convert3x4_4x4(c2ws_est)

    # scale estimated poses to unit sphere
    ts_est = c2ws_est[:, :3, 3]  # (N, 3)

    c2ws_est[:, :3, 3] /= pts_dist_max(ts_colmap)
    c2ws_est[:, :3, 3] *= 2.0


    frustum_length = 0.1
    est_traj_color = np.array([39, 125, 161], dtype=np.float32) / 255
    cmp_traj_color = np.array([249, 65, 68], dtype=np.float32) / 255


    c2ws_est_to_draw_align2cmp = c2ws_est.clone()
    ATE_align=True
    if ATE_align:  # Align learned poses to colmap poses
        c2ws_est_aligned = align_ate_c2b_use_a2b(c2ws_est, c2ws_cmp)  # (N, 4, 4)
        c2ws_est_to_draw_align2cmp = c2ws_est_aligned

        # compute ate
        stats_tran_est, stats_rot_est, _ = compute_ate(c2ws_est_aligned, c2ws_cmp, align_a2b=None)
        print('From est to colmap: tran err {0:.3f}, rot err {1:.2f}'.format(stats_tran_est['mean'],
                                                                             stats_rot_est['mean']))

    frustum_est_list = draw_camera_frustum_geometry(c2ws_est_to_draw_align2cmp.cpu().numpy(), H, W,
                                                    colmap_focal, colmap_focal,
                                                    frustum_length, est_traj_color)
    frustum_colmap_list = draw_camera_frustum_geometry(c2ws_cmp.cpu().numpy(), H, W,
                                                       colmap_focal, colmap_focal,
                                                       frustum_length, cmp_traj_color)

    geometry_to_draw = []
    geometry_to_draw.append(frustum_est_list)
    geometry_to_draw.append(frustum_colmap_list)


    t_est_list = c2ws_est_to_draw_align2cmp[:, :3, 3]
    t_cmp_list = c2ws_cmp[:, :3, 3]


    line_points = torch.cat([t_est_list, t_cmp_list], dim=0).cpu().numpy()  # (2N, 3)
    line_ends = [[i, i+len(image_names)] for i in range(len(image_names))]  # (N, 2) connect two end points.
    # line_color = np.zeros((scene_train.N_imgs, 3), dtype=np.float32)
    # line_color[:, 0] = 1.0

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_ends)
    # line_set.colors = o3d.utility.Vector3dVector(line_color)

    geometry_to_draw.append(line_set)
    o3d.visualization.draw_geometries(geometry_to_draw)


if __name__ == '__main__':
    scene_file='./nerf_llff_data/fern'
    model_file='./myModel.pth'
    vis(scene_file, model_file)
