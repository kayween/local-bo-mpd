# Local Bayesian optimization via maximizing probability of descent

This repository contains code for the paper [Local Bayesian optimization via maximizing probability of descent]().
Our code implementation extends the [GIBO](https://arxiv.org/abs/2106.11899)'s codebase, and more detail can be found in [their repository](https://github.com/sarmueller/gibo).

Please consider citing our paper:
```
@inproceedings{nguyen2022local,
    title = {{Local Bayesian optimization via maximizing probability of descent}},
    author = {Nguyen, Quan and Wu, Kaiwen and Gardner, Jacob R.\ and Garnett, Roman},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2022}
}
```

## Installation
Our implementation relies on mujoco-py 0.5.7 with MuJoCo Pro version 1.31.
To install MuJoCo follow the instructions here: [https://github.com/openai/mujoco-py](https://github.com/openai/mujoco-py).

### Pip
Inside an environment with python 3.8.5 you can install all needed packages with
```
pip install -r requirements.txt
```

### Conda
Or you can create an anaconda environment called mpd using
```
conda env create -f environment.yaml
conda activate mpd
```

### Pipenv
Or you can install and activate and environment via pipenv
```
pipenv install
pipenv shell
```

## Usage
For experiments with synthetic test functions and reinforcement learning problems (e.g. MuJoCo) a command-line interface is supplied.

### Synthetic Test Functions
First generate the needed data for the synthetic test functions.

```
python generate_data_synthetic_functions.py -c ./configs/synthetic_experiment/generate_data_default.yaml
```

Afterwards you can run for instance our method MPD on these test functions.

```
python run_synthetic_experiment.py -c ./configs/synthetic_experiment/mpd_default.yaml -cd ./configs/synthetic_experiment/generate_data_default.yaml
```

### Reinforcement Learning

Run the MuJoCo swimmer environment with the proposed method MPD.

```
python run_rl_experiment.py -c ./configs/rl_experiment/mpd_default.yaml
```

### Custom Objective Functions

Run the Rover trajectory planning function with the proposed method MPD.

```
python run_custom_experiment.py -c ./configs/custom_experiment/mpd_default.yaml
```
