import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time


import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
import socket
import pickle

def d2p(depth_img):
    """
    Convert depth image to point cloud.
    """
    # D435i
    camera_intrinsics = np.array([9.14025757e+02, 9.12897461e+02, 6.36491760e+02, 3.50987793e+02, 7.20000000e+02, 1.28000000e+03, 1.00000005e-03])
    
    # Metaworld
    # camera_intrinsics = np.array([110.85, 110.85, 64, 64, 128, 128, 0.0001])
    

    def get_intrinsics():
        fx = camera_intrinsics[0]
        fy = camera_intrinsics[1]
        cx = camera_intrinsics[2]
        cy = camera_intrinsics[3]
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
    
    def get_depth_scale():
        return camera_intrinsics[6]
    # get camera_intrinsics
    intrinsics = get_intrinsics()
    depth_scale = get_depth_scale()

    # create point cloud
    # st = time.time()
    xmap = np.arange(camera_intrinsics[5])
    ymap = np.arange(camera_intrinsics[4])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth_img * depth_scale
    points_x = (xmap - intrinsics[0,2]) * points_z / intrinsics[0,0]
    points_y = (ymap - intrinsics[1,2]) * points_z / intrinsics[1,1]
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    # et = time.time()
    # print(f"Point cloud generation 1 took {et - st:.4f} seconds")

    # create point cloud
    # st = time.time()
    # depth_frame = frameset.get_depth_frame()
    # pc = rs.pointcloud()
    # points = pc.calculate(depth_frame)
    # vertices = np.asanyarray(points.get_vertices())
    
    # xyz_points = np.array([(v[0], v[1], v[2]) for v in vertices])
    # et = time.time()
    # print(f"Point cloud generation 2 took {et - st:.4f} seconds")

    # error = xyz_points- cloud
    
    return cloud

def farthest_point_sampling(points, num_points=1024, use_cuda=False):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()

        
    else:
        with torch.no_grad():
            points = torch.from_numpy(points)
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.numpy()

    return sampled_points, indices

def color_one_point_cloud(point_cloud, color_image, num_points=1024, use_cuda=False):
    """
    Sample a point cloud to a fixed number of points.
    """
    assert point_cloud.shape[0] == color_image.shape[0] and point_cloud.shape[1] == color_image.shape[1]
    
    y_slice = slice(40, 700)  # Y range
    x_slice = slice(250, 1000)  # X range
    
    # Apply crop to both arrays
    point_cloud = point_cloud[y_slice, x_slice, :]
    color_image = color_image[y_slice, x_slice, :]
    
    # map points and color image to numpy
    points = np.concatenate([point_cloud, color_image], axis=2) # x,y,6

    return points

def sample_one_point_cloud(point_cloud, color_image, num_points=1024, use_cuda=False, crop_scale='middle'):
    """
    Sample a point cloud to a fixed number of points.
    """
    assert point_cloud.shape[0] == color_image.shape[0] and point_cloud.shape[1] == color_image.shape[1]
    
    if crop_scale == 'small':
        y_slice = slice(130, 670)  # Y range
        x_slice = slice(600, 1100)  # X range
    elif crop_scale == 'middle':
        y_slice = slice(130, 670)  # Y range
        x_slice = slice(650, 1250)  # X range
    elif crop_scale == 'large':
        y_slice = slice(10, 710)  # Y range
        x_slice = slice(460, 1200)  # X range
    # Apply crop to both arrays
    point_cloud = point_cloud[y_slice, x_slice, :]
    color_image = color_image[y_slice, x_slice, :]
    
    # map points and color image to numpy
    color_image = np.clip(color_image * 255, 0, 255).astype(np.uint8)
    points = np.concatenate([point_cloud, color_image], axis=2) # x,y,6

    mask = (points[:, :, 2] > 0.0) & (points[:, :, 2] < 2.0) # Valid depth mask
    points = points[mask]

    points = points.reshape(-1, 6)  # Reshape to (N, 6) where N is the number of points

    points_xyz = points[..., :3]  # Extract XYZ coordinates
    points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
    sample_indices = sample_indices.cpu()
    points_rgb = points[sample_indices, 3:][0]

    points = np.hstack((points_xyz, points_rgb))
    return points