'''
this is a copypasta of the same file in the sim repo
https://github.com/abhishekunique/rewardlearning-vid/blob/master/reward_extraction/reward_functions.py
'''

import copy
import functools
import os
from pathlib import Path
import pickle
import time
from typing import List

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.transforms as T
from tqdm import tqdm
import visdom

# from drqv2.replay_buffer import ReplayBuffer
# from drqv2.video import VideoRecorder
from rl.data.replay_buffer import ReplayBuffer
from rl.data.image_buffer import RAMImageReplayBuffer
from r3m import load_r3m

from reward_extraction.models import Policy
from reward_extraction.utils import mixup_criterion, mixup_data
from robot.data import RoboDemoDset
from viz.plot import VisdomVisualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# boiler plate class copy/pasta
class LearnedRewardFunction():
    def __init__(self):
        pass

    def _train_step(self, plot_images: bool = False):
        pass

    def train(self, num_batches):
        for i in tqdm(range(num_batches)):
            if not self.disable_ranking:
                self.ranking_optimizer.zero_grad()
            self.same_traj_optimizer.zero_grad()

            loss_dict = self._train_step(plot_images=(i == 0))

            if not self.disable_ranking:
                loss_dict["ranking_loss"].backward()
                self.ranking_optimizer.step()
            self.running_loss.append(loss_dict["ranking_loss"].item())
            loss_dict["same_traj_loss"].backward()
            self.same_traj_optimizer.step()
            self.running_loss_same_traj.append(loss_dict["same_traj_loss"].item())

            self.train_step += 1

        # do the plot and save thing
        self.train_steps.append(self.train_step)
        self.losses.append(np.mean(self.running_loss))
        self.losses_same_traj.append(np.mean(self.running_loss_same_traj))
        self.losses_std.append(np.std(self.running_loss))
        self.losses_std_same_traj.append(np.std(self.running_loss_same_traj))
        self.running_loss = []
        self.running_loss_same_traj = []
        self.plot_losses()

    def plot_losses(self):
        losses = np.array(self.losses)
        losses_std = np.array(self.losses_std)
        losses_same_traj = np.array(self.losses_same_traj)
        losses_std_same_traj = np.array(self.losses_std_same_traj)

        plt.clf(); plt.cla()
        plt.plot(self.train_steps, self.losses, label="train", color='blue')
        plt.fill_between(
            self.train_steps,
            losses - losses_std,
            losses + losses_std,
            alpha=0.25,
            color='blue'
        )
        plt.legend()
        plt.savefig(f"{self.exp_dir}/training_loss.png")

        plt.clf(); plt.cla()
        plt.plot(self.train_steps, self.losses_same_traj, label="train", color='blue')
        plt.fill_between(
            self.train_steps,
            losses_same_traj - losses_std_same_traj,
            losses_same_traj + losses_std_same_traj,
            alpha=0.25,
            color='blue'
        )
        plt.legend()
        plt.savefig(f"{self.exp_dir}/training_loss_same_traj.png")

        # dump the raw data being plotted
        losses_dict = {
            "loss_ranking": self.losses,
            "loss_ranking_std": self.losses_std,
            "train_iterations": self.train_steps,
            "loss_same_traj": self.losses_same_traj,
            "loss_same_traj_std": self.losses_std_same_traj,
        }
        pickle.dump(losses_dict, open(f"{self.exp_dir}/losses.pkl", "wb"))

    def _calculate_reward(self, x):
        pass

    def save_models(self, save_dir: str = None):
        if save_dir is None:
            if not self.disable_ranking:
                torch.save(self.ranking_network.state_dict(), f"{self.exp_dir}/ranking_policy.pt")
            torch.save(self.same_traj_classifier.state_dict(), f"{self.exp_dir}/same_classifier_policy.pt")
        else:
            if not self.disable_ranking:
                torch.save(self.ranking_network.state_dict(), f"{save_dir}/ranking_policy.pt")
            torch.save(self.same_traj_classifier.state_dict(), f"{save_dir}/same_classifier_policy.pt")


    def load_models(self):
        print(f"loading models from disk in {self.exp_dir}")
        if not self.disable_ranking:
            self.ranking_network.load_state_dict(torch.load(f"{self.exp_dir}/ranking_policy.pt"))
        self.same_traj_classifier.load_state_dict(torch.load(f"{self.exp_dir}/same_classifier_policy.pt"))

        losses_file = f"{self.exp_dir}/losses.pkl"
        if os.path.exists(losses_file):
            losses = pickle.load(open(losses_file, "rb"))
            self.losses = losses["loss_ranking"]
            self.losses_std = losses["loss_ranking_std"]
            self.train_steps = losses["train_iterations"]
            self.losses_same_traj = losses["loss_same_traj"]
            self.losses_std_same_traj = losses["loss_same_traj_std"]
            self.train_step = self.train_steps[-1]

    def eval_mode(self):
        if not self.disable_ranking:
            self.ranking_network.eval()
        self.same_traj_classifier.eval()

    def train_mode(self):
        if not self.disable_ranking:
            self.ranking_network.train()
        self.same_traj_classifier.train()

class RobotLearnedRewardFunction(LearnedRewardFunction):
    def __init__(self,
        obs_size: int,  # if obs size is larger (i.e., ppc was concat'd, ignore)
        exp_dir: str,
        demo_path: str,
        replay_buffer: ReplayBuffer,
        image_replay_buffer: RAMImageReplayBuffer,
        horizon: int,
        train_classify_with_mixup: bool = True,
        add_state_noise: bool = True,
        disable_ranking: bool = False, # GAIL / AIRL
        train_classifier_with_goal_state_only: bool = False, # VICE,.
        load_hdf_into_ram: bool = True,
        obs_is_image: bool = True,
        r3m_net: torch.nn.Module = None,
    ):
        '''
        Learns:
            in (1) /out (0) of distribution - classifier(current_state, goal_state)
            0 to 1 progress - rank(current_state, goal_state)

        Training:
            classifier
                * sample data from demonstrations => 1
                * sample data from replay buffer => 0
            ranking
                * sample data from demonstrations
                    * start with current goal = 0
                    * end with current goal = 1
                    * any state with different goal = 0
        '''
        self.exp_dir = exp_dir
        self.plot_dir = f"{self.exp_dir}/lrf_plots"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir, exist_ok=True)
        self.horizon = horizon
        self.train_classify_with_mixup = train_classify_with_mixup
        self.add_state_noise = add_state_noise
        self.disable_ranking = disable_ranking
        self.obs_size = obs_size
        self.obs_is_image = obs_is_image

        if r3m_net is None:
            self.r3m_net = load_r3m("resnet50") # resnet18
            self.r3m_net.eval()
            self.r3m_net.to("cuda")
        else:
            self.r3m_net = r3m_net

        self.train_classifier_with_goal_state_only = train_classifier_with_goal_state_only
        if self.train_classifier_with_goal_state_only:
            assert self.disable_ranking

        # training parameters
        self.batch_size = 64
        self.transform_batch_size = 8 # apply transform in sets of x images instead of to every image
        assert (self.batch_size % self.transform_batch_size == 0)
        self.lr = 1e-4

        # network definitions
        hidden_depth = 3
        hidden_layer_size = 1024

        if not self.disable_ranking:
            self.ranking_network = Policy(obs_size, 1, hidden_layer_size, hidden_depth)
            self.ranking_network.to(device)
            self.ranking_optimizer = optim.Adam(list(self.ranking_network.parameters()), lr=self.lr)
        self.same_traj_classifier = Policy(obs_size, 1, hidden_layer_size, hidden_depth)
        self.same_traj_classifier.to(device)
        self.same_traj_optimizer = optim.Adam(list(self.same_traj_classifier.parameters()), lr=self.lr)
        self.bce_with_logits_criterion = torch.nn.BCEWithLogitsLoss()

        # make sure there is expert data
        self.expert_data_path = demo_path
        assert os.path.exists(self.expert_data_path)

        if load_hdf_into_ram:
            self.expert_data_ptr = RoboDemoDset(self.expert_data_path, read_only_if_exists=True)
            self.expert_data = [d for d in self.expert_data_ptr]
        else:
            self.expert_data = RoboDemoDset(self.expert_data_path, read_only_if_exists=True)

        self.num_expert_trajs = len(self.expert_data)

        # replay buffer
        self.replay_buffer = replay_buffer
        self.image_replay_buffer = image_replay_buffer
        self._seen_on_policy_data = False

        # bookkeeping
        self.train_step = 0
        self.plot_and_save_frequency = 40
        self.train_steps = []
        self.losses = []
        self.losses_same_traj = []
        self.losses_std = []
        self.losses_std_same_traj = []
        self.running_loss = []
        self.running_loss_same_traj = []

        self.set_image_transforms()

        # debug viz
        vis = visdom.Visdom()
        self.viz = VisdomVisualizer(vis, "main")

        # train the ranking function so it's not giving junk for exploration
        self.init_ranking()


    def init_ranking(self):
        '''
        train the ranking function a bit so it isn't oututting purely 0 during robot exploration
        '''
        ranking_init_losses = []
        num_init_steps = 200
        for i in tqdm(range(num_init_steps)):
            if not self.disable_ranking:
                self.ranking_optimizer.zero_grad()

            ranking_loss, _ = self._train_ranking_step(plot_images=(i % 50 == 0))

            if not self.disable_ranking:
                ranking_loss.backward()
                self.ranking_optimizer.step()
            ranking_init_losses.append(ranking_loss.item())

        # plot and save
        plt.clf(); plt.cla()
        plt.plot([t for t in range(num_init_steps)], ranking_init_losses)
        plt.savefig(f"{self.exp_dir}/ranking_init_loss.png")

    def last_pmr(self):
        return self.progress, self.mask, self.reward

    def _calculate_reward(self, x, dbg: bool = False):
        x = x.to(device)

        with torch.no_grad():
            self.eval_mode()

            mask = torch.sigmoid(self.same_traj_classifier(x)).cpu().numpy()
            if self.disable_ranking:
                reward = mask
            else:
                progress = torch.sigmoid(self.ranking_network(x)).cpu().numpy()
                reward = progress * mask

            self.train_mode()

        if not self._seen_on_policy_data:
            self.mask = 0.0
            self.progress = progress
            self.reward = progress

            # just progress as reward until we've seen on policy data for classifier
            return progress

        self.mask = mask
        self.progress = progress
        self.reward = reward

        if dbg:
            return progress, mask, reward
        else:
            return reward

    def set_image_transforms(self):
        self._train_transforms = T.Compose([
                T.Resize(256),
                T.ColorJitter(brightness=0.5 , contrast=0.1, saturation=0.1, hue=0.3),
                T.RandomRotation(15), # +/-15 degrees of random rotation
                T.RandomCrop(224), # T.CenterCrop(224)
        ])

        self._test_transforms = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
        ])


    def _train_ranking_step(self, plot_images: bool = False):
        '''
        sample from expert data (factuals) => train ranking, classifier positives
        '''
        expert_idxs = np.random.randint(self.num_expert_trajs, size=(self.batch_size,))
        expert_t_idxs = np.random.randint(self.horizon, size=(self.batch_size,))
        expert_other_t_idxs = np.random.randint(self.horizon, size=(self.batch_size,))
        labels = np.zeros((self.batch_size,))
        first_before = np.where(expert_t_idxs < expert_other_t_idxs)[0]
        labels[first_before] = 1.0 # idx is 1.0 if other timestep > timestep

        if self.obs_is_image:
            # get images
            expert_images_t_tensor = torch.cat(
                [T.ToTensor()(self.expert_data[traj_idx]["rgb"][t_idx]).unsqueeze(0) for (traj_idx, t_idx) in zip(expert_idxs, expert_t_idxs)],
                dim=0
            ).to(device)
            expert_other_images_t_tensor = torch.cat(
                [T.ToTensor()(self.expert_data[traj_idx]["rgb"][t_idx]).unsqueeze(0) for (traj_idx, t_idx) in zip(expert_idxs, expert_other_t_idxs)],
                dim=0
            ).to(device)

            # apply data augmentation and convert into form ready for r3m
            expert_processed_images_t_tensor = torch.cat(
                [self._train_transforms(expert_images_t_tensor[i * self.transform_batch_size: (i + 1) * self.transform_batch_size]) for i in range(int(self.batch_size / self.transform_batch_size))]
            )
            expert_processed_images_other_t_tensor = torch.cat(
                [self._train_transforms(expert_other_images_t_tensor[i * self.transform_batch_size: (i + 1) * self.transform_batch_size]) for i in range(int(self.batch_size / self.transform_batch_size))]
            )
            if plot_images:
                self.viz.plot_rgb_batch(expert_processed_images_t_tensor, nrow=self.transform_batch_size, window_name="demo data")

            # convert to r3m vec
            with torch.no_grad():
                expert_states_t_tensor = self.r3m_net(expert_processed_images_t_tensor * 255.0) # r3m expects input to be 0-255
                expert_states_other_t_tensor = self.r3m_net(expert_processed_images_other_t_tensor * 255.0) # r3m expects input to be 0-255
            expert_states_t_np = expert_states_t_tensor.cpu().squeeze().numpy()
            expert_states_other_t_np = expert_states_other_t_tensor.cpu().squeeze().numpy()
        else:
            expert_states_t_np = np.concatenate([self.expert_data[traj_idx]["r3m_vec"][t_idx][None] for (traj_idx, t_idx) in zip(expert_idxs, expert_t_idxs)])
            expert_states_other_t_np = np.concatenate([self.expert_data[traj_idx]["r3m_vec"][t_idx][None] for (traj_idx, t_idx) in zip(expert_idxs, expert_other_t_idxs)])

        if self.add_state_noise:
            expert_states_t_np += np.random.normal(0, 0.01, size=expert_states_t_np.shape)
            expert_states_other_t_np += np.random.normal(0, 0.01, size=expert_states_other_t_np.shape)

        expert_states_t = torch.Tensor(expert_states_t_np).float().to(device)
        expert_states_other_t = torch.Tensor(expert_states_other_t_np).float().to(device)

        ranking_labels = F.one_hot(torch.Tensor(labels).long().to(device), 2).float()

        loss_monotonic = torch.Tensor([0.0])
        if not self.disable_ranking:
            if self.train_classify_with_mixup:
                rank_states = torch.cat([expert_states_t, expert_states_other_t], dim=0)
                rank_labels = torch.cat([ranking_labels[:, 0], ranking_labels[:, 1]], dim=0).unsqueeze(1)

                mixed_rank_states, rank_labels_a, rank_labels_b, rank_lam = mixup_data(rank_states, rank_labels)
                mixed_rank_prediction_logits = self.ranking_network(mixed_rank_states)
                loss_monotonic = mixup_criterion(
                    self.bce_with_logits_criterion, mixed_rank_prediction_logits, rank_labels_a, rank_labels_b, rank_lam
                )
            else:
                expert_logits_t = self.ranking_network(expert_states_t)
                expert_logits_other_t = self.ranking_network(expert_states_other_t)
                expert_logits = torch.cat([expert_logits_t, expert_logits_other_t], dim=-1)

                loss_monotonic = self.bce_with_logits_criterion(expert_logits, ranking_labels)

        return loss_monotonic, expert_states_t

    def _train_step(self, plot_images: bool = False):
        self._seen_on_policy_data = True

        '''
        sample from expert data (factuals) => train ranking, classifier positives
        '''
        loss_monotonic, expert_states_t = self._train_ranking_step()

        '''
        sample from replay buffer data (batch size) (counterfactuals) => train classifier negatives
        '''
        batch = self.replay_buffer.sample(batch_size=self.batch_size)
        image_batch = self.image_replay_buffer.sample(batch_size=self.batch_size)

        if self.obs_is_image:
            # get images and apply data augmentation
            rb_images_tensor = torch.cat(
                [T.ToTensor()(img).unsqueeze(0) for img in image_batch],
                dim=0
            ).to(device)

            # apply data augmentation and convert into form ready for r3m
            rb_processed_images = torch.cat(
                [self._train_transforms(rb_images_tensor[i * self.transform_batch_size: (i + 1) * self.transform_batch_size]) for i in range(int(self.batch_size / self.transform_batch_size))]
            )

            if plot_images:
                self.viz.plot_rgb_batch(rb_processed_images, nrow=self.transform_batch_size, window_name="replay buffer")

            # convert to r3m vec
            with torch.no_grad():
                rb_image_vecs = self.r3m_net(rb_images_tensor * 255.0) # r3m expects input to be 0-255
            rb_cf_states = rb_image_vecs.cpu().squeeze().numpy()
        else:
            rb_cf_states = batch["observations"][:, :self.obs_size]

        if self.add_state_noise:
            rb_cf_states += np.random.normal(0, 0.01, size=rb_cf_states.shape)

        '''
        combine the counterfactals
        '''
        cf_states_np = rb_cf_states
        cf_states = torch.Tensor(cf_states_np).float().to(device)

        classify_states = torch.cat([expert_states_t, cf_states], dim=0)
        traj_labels = torch.cat([torch.ones((expert_states_t.size()[0], 1)), torch.zeros((cf_states.size()[0], 1))], dim=0).to(device)

        if self.train_classify_with_mixup:
            mixed_classify_states, traj_labels_a, traj_labels_b, lam = mixup_data(classify_states, traj_labels)
            mixed_traj_prediction_logits = self.same_traj_classifier(mixed_classify_states)
            loss_same_traj = mixup_criterion(
                self.bce_with_logits_criterion, mixed_traj_prediction_logits, traj_labels_a, traj_labels_b, lam
            )
        else:
            traj_prediction_logits = self.same_traj_classifier(classify_states)
            loss_same_traj = self.bce_with_logits_criterion(traj_prediction_logits, traj_labels)

        return {
            "ranking_loss": loss_monotonic,
            "same_traj_loss": loss_same_traj,
        }


    def eval_lrf(self):
        '''
        replay the lrf over the expert demo states to see if at least that is being set properly
        '''
        train_step_str = str(self.train_step).zfill(7)
        plot_dir = f"{self.plot_dir}/{train_step_str}"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        aggregate_ps, aggregate_ms, aggregate_rs = [], [], []
        for traj_idx, traj in enumerate(self.expert_data):
            rgb_imgs = traj["rgb"] # horizon x embedding_dim
            rgb_img_tensor_batch = torch.cat([T.ToTensor()(rgb).unsqueeze(0) for rgb in rgb_imgs], dim=0)
            processed_imgs = self._test_transforms(rgb_img_tensor_batch)
            with torch.no_grad():
                r3m_vecs = self.r3m_net(processed_imgs * 255.0) # r3m expects input to be 0-255

            progress, mask, reward = self._calculate_reward(r3m_vecs, dbg=True)
            aggregate_ps.append(progress.copy())
            aggregate_ms.append(mask.copy())
            aggregate_rs.append(reward.copy())

            plt.clf(); plt.cla()
            plt.plot(progress, label="progress")
            plt.plot(mask, label='mask')
            plt.plot(reward, label='reward')
            plt.ylim(0, 1)
            traj_idx_str = str(traj_idx).zfill(3)
            plt.legend()
            plt.savefig(f"{plot_dir}/{traj_idx_str}_pmr.png")

        plt.clf(); plt.cla()
        for p in aggregate_ps:
            plt.plot(p)
        plt.ylim(0, 1)
        plt.savefig(f"{plot_dir}/alltrajs_progress.png")

        plt.clf(); plt.cla()
        for p in aggregate_ms:
            plt.plot(p)
        plt.ylim(0, 1)
        plt.savefig(f"{plot_dir}/alltrajs_mask.png")

        plt.clf(); plt.cla()
        for p in aggregate_rs:
            plt.plot(p)
        plt.ylim(0, 1)
        plt.savefig(f"{plot_dir}/alltrajs_rewards.png")


if __name__ == "__main__":
    pass