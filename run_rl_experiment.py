import os
import yaml
import argparse

import numpy as np
import torch
import gym

torch.set_num_threads(1)

from src import config
from src.model import ExactGPSEModel, DerivativeExactGPSEModel
from src.acquisition_function import optimize_acqf_vanilla_bo, optimize_acqf_custom_bo
from src.environment_api import StateNormalizer, EnvironmentObjective, manipulate_reward
from src.policy_parameterizations import MLP, discretize
from src.loop import loop
from src.optimizers import (
    RandomSearch,
    VanillaBayesianOptimization,
    BayesianGradientAscent,
)

import wandb


LOG_WANDB = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run optimization with given optimization method."
    )
    parser.add_argument("-c", "--config", type=str, help="Path to config file.")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # Translate config dictionary.
    cfg = config.insert(cfg, config.insertion_config)

    parameters = {}
    calls = {}
    timesteps_to_reward = {}

    # Usecase 1: optimizing policy for a reinforcement learning environment.
    mlp = MLP(*cfg["mlp"]["layers"], add_bias=cfg["mlp"]["add_bias"])
    len_params = mlp.len_params

    if cfg["mlp"]["discretize"] is not None:
        mlp = discretize(mlp, cfg["mlp"]["discretize"])

    # Evaluate config dictionary (functions etc.).
    cfg = config.evaluate(cfg, len_params)
    if "seed" not in cfg:
        cfg["seed"] = 0

    for trial in range(cfg["trials"]):
        current_seed = cfg["seed"] + trial
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)

        if LOG_WANDB:
            wandb_run = wandb.init(
                project=cfg["wandb_config"]["project_name"],
                entity=cfg["wandb_config"]["entity"],
            )

            wandb_run.name = cfg["wandb_config"]["name"]

        if cfg["mlp"]["state_normalization"]:
            state_norm = StateNormalizer(
                normalize_params=mlp.normalize_params,
                unnormalize_params=mlp.unnormalize_params,
            )
        else:
            state_norm = None

        reward_func = manipulate_reward(
            cfg["mlp"]["manipulate_reward"]["shift"],
            cfg["mlp"]["manipulate_reward"]["scale"],
        )

        objective_env = EnvironmentObjective(
            env=gym.make(cfg["environment_name"]),
            policy=mlp,
            manipulate_state=state_norm,
            manipulate_reward=reward_func,
        )
        objective_env.env.seed(current_seed)

        params, calls_in_iteration = loop(
            params_init=torch.zeros(len_params, dtype=torch.float32),
            max_iterations=cfg["max_iterations"],
            max_objective_calls=cfg["max_objective_calls"],
            objective=objective_env,
            Optimizer=cfg["method"],
            optimizer_config=cfg["optimizer_config"],
            verbose=True,
            wandb_run=wandb_run,
        )

        parameters[trial] = torch.cat(params).numpy()
        calls[trial] = calls_in_iteration
        timesteps_to_reward[trial] = objective_env.timesteps_to_reward

        if LOG_WANDB:
            wandb_run.finish()

    directory = cfg["out_dir"]
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(os.path.join(directory, "parameters"), parameters)
    np.save(os.path.join(directory, "calls"), calls)
    np.save(os.path.join(directory, "timesteps_to_reward"), timesteps_to_reward)
