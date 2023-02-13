# rewardlearning-robot

This codebase interfaces with a Franka Panda robot through [polymetis](https://facebookresearch.github.io/fairo/polymetis/) and runs an implementation of [DroQ](https://arxiv.org/pdf/2110.02034.pdf) from [A Walk in the Park](https://github.com/ikostrikov/walk_in_the_park).

## setup
```
# copy code
git clone git@github.com:dxyang/rewardlearning-robot.git
git submodule init
git submodule update

# create the conda python environment, install most dependencies
conda env create -f environment.yml
pip install --upgrade pip

# install pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# install jax
pip install --upgrade "jax[cuda]==0.4.2" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install walk_in_the_park
cd walk_in_the_park; pip install -e .

# install r3m
cd r3m; pip install -e .
```

## code structure

* `cam` - wrapper to interface with a realsense camera via `pyrealsense2`
* `reward_extraction` - learned reward function model and training code
* `robot` - gym like interface to setup and interact with a robot
* `viz` - some code for visualizing SE3 poses and pointclouds
* `walk_in_the_park` - git submodule. `train_online.py` is the main workhorse that we use. other parts of the codebase also pull from the replay buffer
* `r3m` - git submodule. useful for extracting features from images.

## useful example commands
```
# record demos
python -m robot.record --max_time_per_demo=15 --task test --show_viewer

# train agent
python -m walk_in_the_park.train_online
```

