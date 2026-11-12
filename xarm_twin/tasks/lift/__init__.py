import gymnasium as gym

gym.register(
    id="XArm7-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:XArm7PickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:XArm7PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="XArm7-PickPlace-IK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:XArm7PickPlaceIKEnvCfg"},
    disable_env_checker=True,
)
