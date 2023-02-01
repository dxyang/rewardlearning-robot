import os

import h5py
import numpy as np
from torch.utils.data import Dataset

class RoboDemoDset(Dataset):
    def __init__(self, save_path: str = None, read_only_if_exists: bool = True, should_print: bool = True):
        if os.path.exists(save_path):
            if read_only_if_exists:
                if should_print:
                    print(f"{save_path} already exists! loading this file instead. you will NOT be able to add to it.")
                self.f = h5py.File(save_path, "r")
            else:
                if should_print:
                    print(f"{save_path} already exists! loading this file instead. you will be able to add to it.")
                self.f = h5py.File(save_path, "r+")
            self.length = len(self.f.keys())
            if should_print:
                print(f"{save_path} already has {self.length} trajectories!")
            self.created = False
        else:
            if should_print:
                print(f"creating new dataset at {save_path}")
            self.f = h5py.File(save_path, "w")
            self.length = 0
            self.created = True

        self.save_path = save_path
        self.read_only_if_exists = read_only_if_exists
        self.rgb_shape, self.ja_shape, self.eef_shape = None, None, None

    def __del__(self):
        self.f.close()

    def __len__(self):
        return self.length # this is the number of trajectories stored

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError

        return_dict = {}
        keys = ["rgb", "ja", "eef_pose", "r3m_vec"]

        for k in keys:
            if k in self.f[str(idx)]:
                return_dict[k] = self.f[str(idx)][k][:]

        return return_dict

    def add_traj(self, rgbs, joint_angles, eef_poses):
        if not self.created and self.read_only_if_exists:
            assert False

        # assumes input is in batches
        # (bs, traj_length, rgbs)
        # (bs, traj_length, joint_angles)
        # (bs, eef_poses)
        if self.rgb_shape is None:
            self.rgb_shape = rgbs.shape[1:]
            self.ja_shape = joint_angles.shape[1:]
            self.eef_shape = eef_poses.shape[1:]

        bs = rgbs.shape[0]
        for b_idx in range(bs):
            add_idx = self.length + b_idx
            grp = self.f.create_group(f'{add_idx}')
            grp.create_dataset("rgb", shape=self.rgb_shape, dtype=np.uint8)
            grp.create_dataset("ja", shape=self.ja_shape, dtype=np.float32)
            grp.create_dataset("eef_pose", shape=self.eef_shape, dtype=np.float32)

            grp["rgb"][:] = rgbs[b_idx]
            grp["ja"][:] = joint_angles[b_idx]
            grp["eef_pose"][:] = eef_poses[b_idx]

        self.length += bs
        self.f.flush()

    def _process_rgb_to_r3m_vecs(self):
        # call this to generate r3m vector embeddings for all rgb images in this dataset
        return

if __name__ == "__main__":
    pass