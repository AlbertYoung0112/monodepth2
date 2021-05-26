import kitti_utils
import os
import numpy as np
from scipy import interpolate
from tqdm import tqdm

int_patch_height = 15
# width = 1242
# height = 375

print("Generating Depth Sample")


def depth_interpolation(depth_np):
    height, width = depth_np.shape
    h = int_patch_height
    depth_patch_lb = int((height - int_patch_height) / 2)
    depth_patch_ub = depth_patch_lb + int_patch_height
    hh, ww = np.meshgrid(np.arange(h), np.arange(width), indexing='ij')
    raw_coord = np.stack((hh, ww), axis=2)
    tgt_hh = np.ones(width) * h / 2
    tgt_ww = np.arange(width)
    tgt_coord = np.stack((tgt_hh, tgt_ww), axis=1)
    depth_patch = depth_np[depth_patch_lb:depth_patch_ub, :]
    valid_mask = depth_patch != 0
    valid_depth = depth_patch[valid_mask]
    valid_coord = raw_coord[valid_mask]
    int_depth = interpolate.griddata(valid_coord, valid_depth, tgt_coord, 'nearest')
    if np.any(np.isnan(int_depth)):
        print(len(valid_depth))
    return int_depth


root_dirs = []
for root, dirs, files in os.walk("./kitti_data/2011_09_26/2011_09_26_drive_0001_sync"):
    if root.endswith("sync"):
        root_dirs.append(root)
print(f"Found {len(root_dirs)} Dirs")

for root_dir in tqdm(root_dirs):
    output_dir = [os.path.join(root_dir, "./sample", f"./{i}") for i in range(4)]
    for d in output_dir:
        os.makedirs(d, exist_ok=True)
    pc_files = os.listdir(os.path.join(root_dir, "./velodyne_points/data"))
    calib_path = os.path.join(root_dir, "..")
    for p in pc_files:
        pc_path = os.path.join(root_dir, "./velodyne_points/data", p)
        sample_views = []
        for i in range(4):
            img = kitti_utils.generate_depth_map(calib_path, pc_path, cam=i, vel_depth=False)
            sample = depth_interpolation(img)
            sample_path = os.path.join(output_dir[i], '.'.join(p.split('.')[:-1]))
            np.save(sample_path, sample)
