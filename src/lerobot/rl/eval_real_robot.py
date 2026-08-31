# !/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Standalone real-robot evaluation for HIL-SERL trained policies.

Loads a trained ``gaussian_actor`` checkpoint and rolls it out on the real robot
using the same processor pipeline (teleop intervention + IK) as the actor loop,
but without a learner. Teleop events stay active: press the SpaceMouse RIGHT
button to flag success (reward=1) and LEFT to terminate an episode.

Examples of usage:

```bash
python -m lerobot.rl.eval_real_robot \
    --config_path src/lerobot/configs/train_config_hilserl_so101.json \
    --policy.pretrained_path=outputs/train/hilserl_so101/checkpoints/latest/pretrained_model \
    --eval.n_episodes=10
```

By default the policy mean (deterministic) is executed; use ``--stochastic`` to
sample from the policy distribution.
"""

import logging
import time

import torch
from torch import nn

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import TransitionKey
from lerobot.robots import so_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so_leader, spacemouse  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep

from .gym_manipulator import (
    make_processors,
    make_robot_env,
    reset_and_build_transition,
    step_env_and_process_transition,
)
from .train_rl import TrainRLServerPipelineConfig


def select_action(policy: nn.Module, normalized_observation: dict, stochastic: bool = False) -> torch.Tensor:
    """Select an action: policy mean (default) or a stochastic sample."""
    with torch.no_grad():
        if not stochastic:
            observations_features = None
            if policy.shared_encoder and policy.actor.encoder.has_images:
                observations_features = policy.actor.encoder.get_cached_image_features(normalized_observation)
            _, _, action = policy.actor(normalized_observation, observations_features)
            if policy.config.policy_kwargs.use_tanh_squash:
                action = torch.tanh(action)
        else:
            action = policy.select_action(batch=normalized_observation)

        if policy.config.num_discrete_actions is not None:
            discrete_action = torch.ones((action.shape[0], 1), device=action.device, dtype=action.dtype)
            action = torch.cat([action, discrete_action], dim=-1)
    return action


def rollout_episode(
    cfg: TrainRLServerPipelineConfig,
    env,
    env_processor,
    action_processor,
    policy: nn.Module,
    preprocessor,
    postprocessor,
    stochastic: bool,
) -> tuple[float, bool, int]:
    """Run one episode; returns (episode_reward, success, num_steps)."""
    transition = reset_and_build_transition(env, env_processor, action_processor)
    episode_reward = 0.0
    success = False
    step = 0

    while True:
        start_time = time.perf_counter()

        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }
        normalized_observation = preprocessor.process_observation(observation)
        action = select_action(policy, normalized_observation, stochastic=stochastic)
        action = postprocessor.process_action(action)

        new_transition = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        reward = float(new_transition[TransitionKey.REWARD])
        episode_reward += reward
        info = new_transition[TransitionKey.INFO]
        success = success or bool(info.get(TeleopEvents.SUCCESS, False)) or reward >= 1.0
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)
        transition = new_transition
        step += 1

        if done or truncated:
            success = bool(info.get(TeleopEvents.SUCCESS, False))
            terminate = bool(info.get(TeleopEvents.TERMINATE_EPISODE, False))
            if success:
                end_reason = "SUCCESS (right button reward=1)"
            elif terminate:
                end_reason = "FAILED (left button, reward=0)"
            elif truncated:
                end_reason = "TIMEOUT (time limit, reward=0)"
            else:
                end_reason = "TERMINATED"
            logging.info("[EVAL] Episode end reason: %s", end_reason)
            break

        if cfg.env.fps is not None:
            precise_sleep(max(1 / cfg.env.fps - (time.perf_counter() - start_time), 0.0))

    return episode_reward, success, step


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig, stochastic: bool = False) -> None:
    """Main entry point for real-robot policy evaluation."""
    cfg.validate(allow_existing_output_dir=True)
    assert cfg.policy.pretrained_path is not None, "Please provide --policy.pretrained_path=<checkpoint>"

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.policy.device)

    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy = policy.from_pretrained(str(cfg.policy.pretrained_path))
    policy = policy.to(device).eval()
    assert isinstance(policy, nn.Module)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        dataset_stats=cfg.policy.dataset_stats,
    )

    n_episodes = cfg.eval.n_episodes
    logging.info("Evaluating for %d episodes (%s)", n_episodes, "stochastic" if stochastic else "mean")

    rewards, successes = [], []
    for episode_idx in range(n_episodes):
        episode_reward, success, steps = rollout_episode(
            cfg=cfg,
            env=env,
            env_processor=env_processor,
            action_processor=action_processor,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            stochastic=stochastic,
        )
        rewards.append(episode_reward)
        successes.append(success)
        success_rate = sum(successes) / len(successes)
        logging.info(
            "Episode %d/%d: reward=%.2f success=%s steps=%d (success rate %.0f%%)",
            episode_idx + 1,
            n_episodes,
            episode_reward,
            success,
            steps,
            100 * success_rate,
        )

    logging.info(
        "Final success rate: %.0f%% (%d/%d)", 100 * sum(successes) / n_episodes, sum(successes), n_episodes
    )
    env.close()


if __name__ == "__main__":
    import sys

    stochastic = "--stochastic" in sys.argv
    if stochastic:
        sys.argv.remove("--stochastic")
    main(stochastic=stochastic)
