import gymnasium as gym

gym.register(
    id="XArm6-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:XArm6PickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:XArm6PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="XArm6-PickPlace-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:XArm6PickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:XArm6PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="XArm6-PickPlace-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:XArm6PickPlaceIKEnvCfg"},
    disable_env_checker=True,
)
