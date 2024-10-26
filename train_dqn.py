import getpass
import logging
import os
from functools import partial
from pathlib import Path

import torch.multiprocessing as mp
from fastai.callback.schedule import SchedCos, SchedExp, combine_scheds
from fastcore.foundation import L
from fastcore.script import call_parse

from tmenv.gbx import campaigns_maps_uid, official_campaigns
from tmenv.tmenv import DiscreteActionHandler, Tmenv
from tmenv.wrappers import (
    EpisodeInfoWrapper,
    RotationEnvWrapper,
    TelemetryState,
    TrajectoryDoneWrapper,
    TrajectoryEnvWrapper,
    TrajectoryState,
    TrajectoryTravelRewardWrapper,
    VisualState,
    WheelState,
    AlignedState,
)
from tmlearn.agent.dqn import (
    DqnAgent,
    EpsilonGreedyExploration,
    NoExploration,
    RankSoftmaxExploration,
)
from tmlearn.collector import (
    ExplorationSwitcherCallback,
    IgnoreLaggyStepCallback,
    IgnoreRespawnAndUnfinished,
    LagCallback,
    MemoryFullRecoverCallback,
    SessionScheduler,
    TelemetryTimeoutRecoverCallback,
)
from tmlearn.neural import (
    init_weights,
    Conv1dFlat,
    DuelingNet,
    Flatten,
    StateCatNet,
    VisualNet,
    TimmModel,
)
from tmlearn.replay_buffer import PrioritizedReplayBuffer
from tmlearn.training import (
    ExperienceTimeoutCallback,
    LrSchedulerCallback,
    MapScheduler,
    Training,
)

mp.set_start_method("spawn", force=True)
logging.basicConfig(
    format="[%(asctime)s][%(process)d][%(levelname)s]: %(message)s",
    datefmt="%d-%b-%y][%H:%M:%S",
    level=logging.DEBUG,
)


@call_parse
def main(
    g: str = None,  # wandb group to load
    a: str = "agent-latest.pth",  # wandb agent to load
):
    if g is None:
        answer = input("You're starting a new training. Continue? (y/n): ")
        if answer.lower() not in ["y", "yes"]:
            print("Exiting.")
            return
        print("Continuing...")

    tmwine = Path(f"/home/{getpass.getuser()}/Development/tmwine")
    device = os.getenv("TMENV_DEVICE", "cuda:0")

    timestep = 0.1
    discount_factor = 0.95

    def make_env(name, window_x, window_y):
        env = Tmenv(
            name=name,
            width=960,
            height=540,
            prefix_path=tmwine / name,
            prefix_template_path=tmwine / "template",
            credential_path=tmwine / "credentials" / f"{name}.user.dat",
            timestep=timestep,
            action_handler=DiscreteActionHandler(),
            device=device,
            window_x=window_x,
            window_y=window_y,
        )
        env = RotationEnvWrapper(env)
        env = TrajectoryEnvWrapper(env, minimum_speed=7, max_distance=50)
        env = TrajectoryDoneWrapper(
            env,
            max_mistake_duration=4,
            respawn_sequence=["launched", "standing"] * 2,
            give_up_before_first_cp=False,
        )
        env = TrajectoryTravelRewardWrapper(
            env,
            mistake_reward=-50 / 3.6,
            discount_factor=discount_factor,
            penalty_threshold=10.0,
        )
        env = VisualState(env, shape=(270, 480))
        env = TrajectoryState(env, nb_positions=8, spacing=50)
        env = AlignedState(env)
        env = WheelState(env)
        env = EpisodeInfoWrapper(env)
        env = TelemetryState(env)
        return env

    envs_args = L(("tmai1", 0, 0), ("tmai2", 0, 540), ("tmai3", 960, 0))
    envs = envs_args.starmap(make_env)

    maps_uid = campaigns_maps_uid(
        official_campaigns()[:13], slice(15)
    )  # White, green and blue maps up to Summer 2023 = 195 maps

    obs_space = envs[0].observation_space
    model = DuelingNet(
        model=StateCatNet(
            visual=TimmModel("regnetx_002", input_dim=obs_space["visual"].shape, positional=True),
            telemetry=Flatten(obs_space["telemetry"].shape),
            trajectory=Flatten(obs_space["trajectory"].shape),
            aligned=Flatten(obs_space["aligned"].shape),
            wheel=Conv1dFlat(obs_space["wheel"].shape, hidden=16),
        ),
        hidden=[512, 256],
        output_dim=envs[0].action_space.n,
        p=0.1,
    )
    model.apply(init_weights)

    agent = DqnAgent(
        model,
        exploration=RankSoftmaxExploration(),
        device=device,
        gamma=discount_factor,
    )
    replay_buffer = PrioritizedReplayBuffer(capacity=2*4096, device=device)

    collector_cbs = [
        partial(TelemetryTimeoutRecoverCallback, 0.5),
        partial(MemoryFullRecoverCallback, 94.0),
        LagCallback,
        partial(IgnoreLaggyStepCallback, timestep * 2),
        partial(
            ExplorationSwitcherCallback,
            [
                EpsilonGreedyExploration(0.2),
                RankSoftmaxExploration(0.9),
                EpsilonGreedyExploration(0.1),
                RankSoftmaxExploration(1.0),
                EpsilonGreedyExploration(0.05),
                RankSoftmaxExploration(1.1),
                EpsilonGreedyExploration(0.02),
                EpsilonGreedyExploration(0.01),
                NoExploration(),
                NoExploration(),
            ],
        ),
        IgnoreRespawnAndUnfinished,
    ]

    base_lr = 3e-4
    training_cbs = [
        ExperienceTimeoutCallback(5 * 60),
        LrSchedulerCallback(
            combine_scheds(
                [0.01/103.01, 3/103.01, 100/103.01],
                [SchedCos(base_lr/100, base_lr), SchedCos(base_lr, base_lr/10), SchedExp(base_lr/10, base_lr/1000)]
            ), max_time=103.01 * 24 * 3600
        ),
    ]

    training = Training(
        map_scheduler=MapScheduler(maps_uid),
        envs=envs,
        agent=agent,
        replay_buffer=replay_buffer,
        session_scheduler=SessionScheduler(nb_episode=10),
        collector_cbs=collector_cbs,
        training_cbs=training_cbs,
        batch_size=32,
        burnin=2048,
        wandb_project=os.getenv("TMLEARN_WANDB_PROJECT"),
        wandb_entity=os.getenv("TMLEARN_WANDB_ENTITY"),
        wandb_load_group=g,
        wandb_load_agent=a,
    )
    training.fit()