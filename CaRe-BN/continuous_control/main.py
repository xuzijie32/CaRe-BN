import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
from utils import ReplayBuffer,eval_policy
from actor_critic import DDPG, TD3

'''Implementation of CaRe-BN'''


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--spiking_neurons", default="DN")  # Spiking neurons (LIF, CLIF, DN or ANN)
    parser.add_argument("--care_bn", default=True)  # Whether to use the CaRe-BN
    parser.add_argument("--recalibration_freq", default=5e3, type=int)  # How often (time steps) we re-calibrate BN statistics
    parser.add_argument("--recalibration_batchs", default=100, type=int)  # Numbers of re-calibration batchs
    parser.add_argument("--policy", default="TD3")  # RL algorithm (DDPG or TD3)
    parser.add_argument("--env", default="Hopper-v4")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_trainsteps", default=1e6 , type=int)  # Max training steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    # Paras for TD3:
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    args = parser.parse_known_args()[0]


    file_name = f"{args.spiking_neurons}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.spiking_neurons}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "BN": args.care_bn,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "spiking_neurons": args.spiking_neurons,
        "recalibration_batchs": args.recalibration_batchs
    }

    if args.policy == "TD3":
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs["policy_noise"] = args.policy_noise * max_action
        policy = TD3(**kwargs)

    elif args.policy == "DDPG":
        policy = DDPG(**kwargs)
    else:
        raise ValueError('Policy can only be DDPG and TD3')

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    state = state[0]
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_trainsteps + args.start_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done1, done2, _ = env.step(action)
        done = done1 + done2
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            state = state[0]
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % args.recalibration_freq == 0:
            policy.re_calibration(replay_buffer)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)

    policy.save(f"./models/{file_name}")