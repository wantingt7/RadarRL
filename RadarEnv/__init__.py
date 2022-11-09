from gym.envs.registration import register

# update jammer policy and reward only take the last pause
# old jammer policy: jammer观测时间长，可以选择等待观察，不做动作
register(
    id='RadarEnv-v0',
    entry_point='RadarEnv.RadarEnv:RadarEnv',
    max_episode_steps=500,
    reward_threshold=500.0,
)

register(
    id='RadarEnvModel-v0',
    entry_point='RadarEnv.RadarEnvModel:RadarEnvModel',
    max_episode_steps=500,
    reward_threshold=500.0,
)

# for one version of model ensemble
register(
    id='RadarEnvModel-v1',
    entry_point='RadarEnv.RadarEnvModel_v2:RadarEnvModel',
    max_episode_steps=500,
    reward_threshold=500.0,
)

# new jammer policy: jammer观测时间短
register(
    id='RadarEnv-v2',
    entry_point='RadarEnv.RadarEnv_new:RadarEnv',
    max_episode_steps=500,
    reward_threshold=500.0,
)

register(
    id='RadarEnvModel-v2',
    entry_point='RadarEnv.RadarEnvModel_new:RadarEnvModel',
    max_episode_steps=500,
    reward_threshold=500.0,
)

# reward不是pd，而是未被干扰个数
register(
    id='RadarEnv-v3',
    entry_point='RadarEnv.RadarEnv_simple:RadarEnv',
    max_episode_steps=500,
    reward_threshold=500.0,
)

register(
    id='RadarEnvModel-v3',
    entry_point='RadarEnv.RadarEnvModel_simple:RadarEnvModel',
    max_episode_steps=500,
    reward_threshold=500.0,
)