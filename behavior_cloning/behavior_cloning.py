import numpy as np
import torch
import torch.nn as nn
import torch.functional as F 
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from robot.data import RoboDemoDset
from robot.vr import OculusController
from robot.xarm_env import XArmCentimeterSafeEnvironment
from reward_extraction.models import MLP
import sys
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import psutil
import collections
import select
import tty
import termios

from robot.data import RoboDemoDset
from robot.utils import HZ
from robot.vr import OculusController
from robot.xarm_env import XArmCentimeterSafeEnvironment

DEMO = "tissue2"

# export PYTHONPATH="${PYTHONPATH}:/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class BCNet(nn.Module):
    def __init__(self) -> None:
        super(BCNet, self).__init__()
        self.fc1 = nn.Linear(4, 4000)
        self.fc2 = nn.Linear(4000, 4000)
        self.fc3 = nn.Linear(4000, 4)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)    
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
def test_data():
    expert_data = RoboDemoDset(f'/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/{DEMO}/demos.hdf', read_only_if_exists=True)
    data = expert_data.__getitem__(13)
    deltas = data['ja']
    env = XArmCentimeterSafeEnvironment(
            control_frequency_hz=HZ,
            use_camera=False,
            # TODO: create some sort of button pressing mechanism to open and close the gripper,
            use_gripper=True,
            random_reset_home_pose=False,
            speed=150,
            low_colision_sensitivity=True,
            scale_factor=10
        )
    env.reset()

    for delta in deltas:
        env.step(delta, delta=True)


def data(SCALE) -> Tuple[DataLoader, DataLoader]:
    expert_data = RoboDemoDset(f'/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/{DEMO}/demos.hdf', read_only_if_exists=True)
   
    

    state = []
    targets = []

    trajs = []
    for traj in expert_data:
        trajs.append(trajs)
        ee_poses = traj['eef_pose']
        # VERY BAD WE KNOW
        deltas = traj['ja']
        # delta_targets = [] # np.zeros((len(joint_poses)-1, 3))
        for i in range(1, len(ee_poses)-1):
            # if np.sum(joint_poses[i+1] - joint_poses[i]) == 0:
            #     continue
            state.append(ee_poses[i-1] / SCALE)
            # targets.append(deltas[i])
            # cur_pose = joint_poses[i+1][:3] - joint_poses[i][:3] # this is the xyz's
            # cur_pose = np.append(cur_pose, joint_poses[i][3]) # this is the gripper
            targets.append(deltas[i])
            # targets.append(joint_poses[i]/100)

    state, targets = torch.Tensor(np.array(state)), torch.Tensor(np.array(targets))

    perm = torch.randperm(len(state))
    state, targets = state[perm], targets[perm]

    test_idx = int(len(state) * 1.0)
    train_x, train_y = state[:test_idx], targets[:test_idx]
    # test_x, test_y = state[test_idx:], targets[test_idx:]

    train_loader = DataLoader(BCDataset(train_x, train_y), batch_size=32, shuffle=True)
    # test_loader = DataLoader(BCDataset(test_x, test_y), batch_size=128, shuffle=True)

   
    return train_loader, None #test_loader



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
        # print(f"Current State: {data}")
        # print(f"To predict: {labels}")
        optimizer.zero_grad()
        preds = net(data)
        # print(f"Error: {preds - labels}")
        loss = critirion(preds, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print(f'Average train loss is {np.mean(losses)}')
    return np.mean(losses)

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

        return np.mean(losses)
    
def hyp_search(epoch_number):
    train_loader, test_loader = data(1)
    lrs = [0.01, 0.001, 0.0001, 0.00001]
    wds = [0, 0.001, 0.0001, 0.00001, 0.000001] 
    eps = [50, 100, 150, 200, 500]
    best = 1000
    best_comb = [0, 0, 0]
    for lr in lrs:
        for wd in wds:
            for ep in eps:
                net = BCNet().to(device)
                optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd) # best hyper: nonlinear = LReLU, lr = 0.00001, decay = 0.01
                epochs = ep
                train_losses = []
                valid_losses = []
                for _ in range(epochs): 
                    train_loss = one_train_iter(net, optimizer, train_loader)
                    valid_loss = eval(net, test_loader)
                    train_losses.append(train_loss)
                    valid_losses.append(valid_loss)
                cur_loss = np.mean(valid_losses)
                if cur_loss < best:
                    best = cur_loss
                    best_comb[0] = lr
                    best_comb[1] = wd
                    best_comb[2] = ep
    print(f"\nBest: {best}")
    print(f"Best Combo: {best_comb}")

def try_until_estop(model, SCALE):
    env = XArmCentimeterSafeEnvironment(
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
    scale = float(sys.argv[2])
    if mode == "train":
        epoch_num = int(sys.argv[3])
        train_loader, test_loader = data(scale)
        
        mse_loss = nn.MSELoss()
        net = BCNet().cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)
        for j in range(epoch_num):
            for batch in train_loader:
                obs, act = batch
                obs = obs.cuda()
                act = act.cuda()
                act_pred = net(obs)
                
                loss = torch.sum((act_pred - act)**2, dim=-1).mean(dim=0)

                # loss = mse_loss(act_pred, act)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if j % 100 == 0:
                expert_data = RoboDemoDset(f'/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/{DEMO}/demos.hdf', read_only_if_exists=True)
                obs = expert_data[0]['eef_pose']
                act = expert_data[0]['ja']
                obs_torch = torch.from_numpy(obs).cuda()
                out_ac = net(obs_torch)
                fig, ax = plt.subplots(4, 1)
                out_ac_np = out_ac.detach().cpu().numpy()
                for j in range(4):
                    ax[j].plot(out_ac_np[:, j], color='red')
                    ax[j].plot(act[:, j], color='blue')
                plt.savefig('test_bc_cur_progress.png')

            print("Epoch %d loss is %f"%(j, loss.item()))
            
        expert_data = RoboDemoDset(f'/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/{DEMO}/demos.hdf', read_only_if_exists=True)
        obs = expert_data[0]['eef_pose']
        act = expert_data[0]['ja']
        obs_torch = torch.from_numpy(obs).cuda()
        out_ac = net(obs_torch)
        fig, ax = plt.subplots(4, 1)
        out_ac_np = out_ac.detach().cpu().numpy()
        for j in range(4):
            ax[j].plot(out_ac_np[:, j], color='red')
            ax[j].plot(act[:, j], color='blue')
        plt.savefig('test_bc_pred1.png')
        torch.save(out_ac, "out_ac.pt")
        # print(len(train_loader))
        # # net = MLP(3, 300, 3, 3, nn.LeakyReLU()).to(device)
        # net = BCNet().to(device)
        # optimizer = torch.optim.Adam(net.parameters()) # 0.0001 0.001
        # print(len(train_loader))
        # epochs = epoch_num
        # train_losses = []
        # valid_losses = []
        # for _ in range(epochs): 
        #     train_loss = one_train_iter(net, optimizer, train_loader)
        #     valid_loss = eval(net, test_loader)
        #     train_losses.append(train_loss)
        #     valid_losses.append(valid_loss)
        # for state, labels in test_loader:
        #     state, labels = state.to(device), labels.to(device)
        #     # print(state)
        #     print(net(state))
        #     print(labels)
        #     # print(state.shape)
        #     # print(labels.shape)
        #     # print(net(state).shape)
        #     break

        torch.save(net, f"/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/behavior_cloning/{DEMO}.pt")
        print("saved, terminate.")

        # print("plotting training data")
        # plt.plot(train_losses, label='train')
        # plt.plot(valid_losses, label='validation')
        # plt.legend()
        # plt.show()

        # import IPython
        # IPython.embed()

        # try_until_estop(net, 1)
    elif mode == "deploy":
        deploy(scale)
    elif mode == "test_data":
        data(1)
    elif mode == "search":
        epoch_num = int(sys.argv[3])
        hyp_search(epoch_num)
    elif mode == "predict":
        traj_num = int(sys.argv[3])
        pred_replay(traj_num)
    else: 
        print("Wrong argument, please enter either 'train' or 'deploy'")
    
# def debug():

def pred_replay(num):
    expert_data = RoboDemoDset(f'/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/data/demos/{DEMO}/demos.hdf', read_only_if_exists=True)
    net = torch.load(f"/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/behavior_cloning/{DEMO}.pt")
    obs = expert_data[num]['eef_pose']
    act = expert_data[num]['ja']
    obs_torch = torch.from_numpy(obs).cuda()
    out_ac = net(obs_torch)
    fig, ax = plt.subplots(4, 1)
    out_ac_np = out_ac.detach().cpu().numpy()
    for j in range(4):
        ax[j].plot(out_ac_np[:, j], color='red')
        ax[j].plot(act[:, j], color='blue')
    plt.savefig(f'test_bc_pred{num}.png')
    
    env = XArmCentimeterSafeEnvironment(
        control_frequency_hz=HZ,
        use_camera=False,
        # TODO: create some sort of button pressing mechanism to open and close the gripper,
        use_gripper=True,
        random_reset_home_pose=False,
        speed=150,
        low_colision_sensitivity=True,
        scale_factor=10
    )
    net.eval()
    net.double()
    while True:
            obs = env.reset()

            # preds = torch.load("out_ac.pt")
            print('press "a" then "enter" to start, hit "a" again to stop')
            console = input()
            a = console == 'a'
            d = console == 'd'
            while not a and not d:
                console = input()
                a = console == 'a'
                d = console == 'd'

            if d:
                break

            def isData():
                return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                with torch.no_grad():
                    while True:
                        # Get Button Input (only if True) --> handle extended button press...

                        # Terminate...
                        if isData():
                            c = sys.stdin.read(1)
                            print(c)
                            if c == 'a':  # x1b is ESC
                                print("\tHit (a) stop deploying\n")
                                break

                        for pred in out_ac: 
                            act = np.array(pred.cpu())
                            env.step(action=act, delta=True)
                        
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            env.close()


    
def deploy(SCALE):
    net = torch.load(f"/home/xarm/Documents/JacobAndDavin/test/rewardlearning-robot/behavior_cloning/{DEMO}.pt")
    # oculus = OculusController()
    env = XArmCentimeterSafeEnvironment(
        control_frequency_hz=HZ,
        use_camera=False,
        # TODO: create some sort of button pressing mechanism to open and close the gripper,
        use_gripper=True,
        random_reset_home_pose=False,
        speed=150,
        low_colision_sensitivity=True,
        scale_factor=10
    )
    net.eval()
    net.double()
    while True:
            obs = env.reset()

            # preds = torch.load("out_ac.pt")
            print('press "a" then "enter" to start, hit "a" again to stop')
            console = input()
            # print(console)
            a = console == 'a'
            d = console == 'd'
            while not a and not d:
                console = input()
                a = console == 'a'
                d = console == 'd'

            if d:
                break

            def isData():
                return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                with torch.no_grad():
                    while True:
                        # Get Button Input (only if True) --> handle extended button press...

                        # Terminate...
                        if isData():
                            c = sys.stdin.read(1)
                            print(c)
                            if c == 'a':  # x1b is ESC
                                print("\tHit (a) stop deploying\n")
                                break

                        # Otherwise no termination, keep on recording...
                        
                        pos = torch.tensor(obs['ee_pos']).double().to(device)
                        update = np.array(net(pos).cpu())
                        # if isData():
                        #     c = sys.stdin.read(1)
                        #     print(c)
                        #     if c == 's':
                        #         update[3] = 1
                        #         print("here")
                        obs, _, _, _ = env.step(action=update, delta=True)
                        # print(update[3])

                        # for pred in preds: 
                        #     act = np.array(pred.cpu())
                        #     env.step(action=act, delta=True)
                        
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            env.close()
          
            


if __name__ == '__main__':
    main()




































