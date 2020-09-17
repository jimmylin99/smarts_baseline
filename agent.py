# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CHECK load_path, and ./model_submission/ directory
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
from pathlib import Path

from common.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

from common.saved_model import RLlibTFCheckpointPolicy

load_path = "model_submission/checkpoint_60/checkpoint-60"


# load saved model
# NOTE: the saved model includes only one policy
policy_handler = RLlibTFCheckpointPolicy(
    Path(__file__).parent / load_path,
    "PPO",
    "default_policy",
    OBSERVATION_SPACE,
    ACTION_SPACE,
)

# Agent specs in your submission must be correlated to each scenario type, in other words, one agent spec for one scenario type.
# DO NOT MODIFY THIS OBJECT !!!
scenario_dirs = [
    "crossroads",
    "double_merge",
    "ramp",
    "roundabout",
    "t_junction",
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = agent_spec
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`
    agent_specs[k].policy_builder = lambda: policy_handler

__all__ = ["agent_specs"]
