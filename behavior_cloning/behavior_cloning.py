import numpy as np
import torch
import torch.nn as nn
import torch.functional as F 
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from robot.data import RoboDemoDset
from robot.base_robot import XArmTaskSpaceEnv
from reward_extraction.models import MLP
import sys
# export PYTHONPATH="${PYTHONPATH}:/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class BCNet(nn.Module):
    def __init__(self) -> None:
        super(BCNet, self).__init__()
        self.fc1 = nn.Linear(3, 30)
        self.fc2 = nn.Linear(30, 300)
        self.fc4 = nn.Linear(300, 30)
        self.fc3 = nn.Linear(30, 3)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)    
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
    

def data(SCALE) -> Tuple[DataLoader, DataLoader]:
    expert_data = RoboDemoDset('/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/test/demos.hdf', read_only_if_exists=True)
    state = []
    targets = []
    for traj in expert_data:
        joint_poses = traj['eef_pose']
        # delta_targets = [] # np.zeros((len(joint_poses)-1, 3))
        for i in range(0, len(joint_poses) -1):
            if np.sum(joint_poses[i+1] - joint_poses[i]) == 0:
                continue
            state.append(joint_poses[i] / SCALE)
            targets.append(joint_poses[i+1] / SCALE - joint_poses[i] / SCALE)
            # targets.append(joint_poses[i]/100)
        # print(joint_poses[:10])
        # print(delta_targets[:10])

        # targets += joint_poses[:-1] - joint_poses[1:]
        # state += joint_poses[:-1]

        # targets.append(targets)
        
    state = np.array(state)
    targets = np.array(targets)
    perm = torch.randperm(len(state))
    state = torch.Tensor(state)
    targets = torch.Tensor(targets)
    state = state[perm]
    targets = targets[perm]
    test_idx = int(len(state) * 0.8)
    train_x = state[:test_idx]
    train_y = targets[:test_idx]
    test_x = state[test_idx:]
    test_y = targets[test_idx:]

    train_loader = DataLoader(BCDataset(train_x, train_y), batch_size=128, shuffle=True)
    test_loader = DataLoader(BCDataset(test_x, test_y), batch_size=128, shuffle=True)
    return train_loader, test_loader



class BCDataset(Dataset):
    def __init__(self, x, y) -> None:
        # super().__init__()
        self.x = x
        self.y = y
    def __len__(self):
        return(len(self.x))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def one_train_iter(net, optimizer, train_loader):
    critirion = nn.MSELoss()
    net.train()
    losses = []
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = net(data)
        loss = critirion(preds, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f'Average train loss is {np.mean(losses)}')
    return (np.mean(losses))

def eval(net, test_loader):
    critirion = nn.MSELoss()
    with torch.no_grad():
        losses = []
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            preds = net(data)
            # print(preds)
            # print(labels)
            loss = critirion(preds, labels)
            loss = loss.item()
            losses.append(loss)
        print(f'Average test loss is {np.mean(losses)}')


def try_until_estop(model, SCALE):
    env = XArmTaskSpaceEnv(
        control_frequency_hz=10,
        use_camera=True,
        # TODO: create some sort of button pressing mechanism to open and close the gripper,
        random_reset_home_pose=False
    )
    obs = env.reset()
    model.eval()
    model.double()
    with torch.no_grad():
        while(True):
            pos = torch.tensor(obs['ee_pos'] / SCALE).double().to(device)
            print(pos)
            update = np.array(model(pos).cpu())
            print(update * SCALE)
            obs = env.step(action=(update*SCALE), delta=True)[0]

def main():
    mode = sys.argv[1]
    scale = int(sys.argv[2])
    if mode == "train":
        epoch_num = int(sys.argv[3])
        train_loader, test_loader = data(scale)
        # net = MLP(3, 300, 3, 3, nn.LeakyReLU()).to(device)
        net = BCNet().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.00001, weight_decay=0.01) # best hyper: nonlinear = LReLU, lr = 0.00001, decay = 0.01
        print(len(train_loader))
        epochs = epoch_num
        for _ in range(epochs): 
            one_train_iter(net, optimizer, train_loader)
            eval(net, test_loader)
        for state, labels in test_loader:
            state, labels = state.to(device), labels.to(device)
            # print(state)
            print(net(state))
            print(labels)
            # print(state.shape)
            # print(labels.shape)
            # print(net(state).shape)
            break

        # save model
        print("saving model...")
        torch.save(net, "behavior_cloning/bc_net2.pt")
        print("saved, terminate.")Xa
        # try_until_estop(net, 1)
    elif mode == "deploy":
        deploy()
    else: 
        print("Wrong argument, please enter either 'train' or 'deploy'")
    
# def debug():
    
def deploy():
    net = torch.load("behavior_cloning/bc_net2.pt")
    try_until_estop(net, 1)


if __name__ == '__main__':
    main()




































