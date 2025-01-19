import argparse
import os
import time
import numpy as np
import cv2

import fusion
from data_loader import readGaustudioColmap, read_depth_meter


def tsdf_fusion(input_path, output_path):
    # ======================================================================================================== #
    # load colmap data from gaustudio
    # ======================================================================================================== #
    cam_infos, cam_intr = readGaustudioColmap(input_path)
    # ======================================================================================================== #

    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    print("Estimating voxel volume bounds...")
    vol_bnds = np.zeros((3, 2))
    for cam_info in cam_infos:
        depth_im = read_depth_meter(cam_info.depth_path)

        T = cam_info.T.reshape(3, 1)
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = cam_info.R
        cam_pose[:3, 3] = T.squeeze()

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)
    print("Initialize Success!")

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    i = 0
    n_imgs = len(cam_infos)
    for cam_info in cam_infos:
        print("Fusing frame %d/%d" % (i + 1, n_imgs))
        i += 1

        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(cv2.imread(cam_info.image_path), cv2.COLOR_BGR2RGB)
        depth_im = read_depth_meter(cam_info.depth_path)

        T = cam_info.T.reshape(3, 1)
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = cam_info.R
        cam_pose[:3, 3] = T.squeeze()

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Save result
    # ======================================================================================================== #
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply and trimesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite(os.path.join(output_path, "mesh.ply"), verts, faces, norms, colors)
    fusion.meshwrite_trimesh(os.path.join(output_path, "trimesh.ply"), verts, faces)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite(os.path.join(output_path, "pc.ply"), point_cloud)
    # ======================================================================================================== #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, help='Path to input data')
    parser.add_argument('--output_path', required=True, help='Path to output result')
    args = parser.parse_args()

    tsdf_fusion(args.input_path, args.output_path)
