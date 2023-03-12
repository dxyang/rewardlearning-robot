import os

import h5py
import numpy as np
import torch
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

            # print(f"rgbs: {rgbs}")
            print(f"b_idx: {b_idx}")
            grp["rgb"][:]
            grp["rgb"][:] = rgbs[b_idx]
            grp["ja"][:] = joint_angles[b_idx]
            grp["eef_pose"][:] = eef_poses[b_idx]

        self.length += bs
        self.f.flush()

    def _process_rgb_to_r3m_vecs(self):
        # call this to generate r3m vector embeddings for all rgb images in this dataset
        from r3m import load_r3m
        import torchvision.transforms as T

        r3m_net = load_r3m("resnet50") # resnet18
        transforms = T.Compose([
            T.ToTensor(), # divides by 255, will also convert to chw
            T.Resize(256),
            T.CenterCrop(224),
        ])
        torch_device = "cuda"
        r3m_net.eval()
        r3m_net.to(torch_device)

        for traj_idx in range(self.length):
            imgs = self.f[str(traj_idx)]["rgb"][:]

            img_tensors = [transforms(img).unsqueeze(0) for img in imgs]
            img_batch = torch.cat(img_tensors, axis=0).to(torch_device)
            with torch.no_grad():
                embeddings_batch = r3m_net(img_batch * 255.0)

            embeddings_batch_np = embeddings_batch.cpu().numpy()
            grp = self.f[str(traj_idx)]
            if "r3m_vec" in grp:
                del grp["r3m_vec"]
            grp.create_dataset("r3m_vec", shape=embeddings_batch_np.shape, dtype=np.float32)
            grp["r3m_vec"][:] = embeddings_batch_np

        return

if __name__ == "__main__":
    from tap import Tap
    from pathlib import Path

    class ArgumentParser(Tap):
        task: str                                           # Task ID for demonstration collection

    args = ArgumentParser().parse_args()
    demo_root = Path.cwd() / "data/demos"
    data_path = demo_root / args.task / "demos.hdf"

    dset = RoboDemoDset(save_path=data_path, read_only_if_exists=False)
    dset._process_rgb_to_r3m_vecs()
    import pdb; pdb.set_trace()