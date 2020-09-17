import argparse
from pathlib import Path

import gym
import time
import matplotlib.pyplot as plt
import matplotlib

from common.continuous_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE

# from utils.discrete_space import agent_spec, OBSERVATION_SPACE, ACTION_SPACE
from common.saved_model import RLlibTFCheckpointPolicy
from common.utils import get_submission_num


def parse_args():
    parser = argparse.ArgumentParser("run simple keep lane agent")
    # env setting
    parser.add_argument("--scenario", "-s", type=str, help="Path to scenario")
    parser.add_argument("--load_path", "-p", type=str, help="path to stored model")
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )

    return parser.parse_args()


def main(args):
    # scenario_path = Path(args.scenario).absolute()

    scenario_path = Path(args.scenario).absolute()
    n_mission = get_submission_num(scenario_path)  # here is 4 for dataset_public/crossroads/2lane/

    if n_mission == -1:
        raise ValueError("No mission can be found")

    agent_spec.policy_builder = lambda: RLlibTFCheckpointPolicy(
        Path(args.load_path).absolute(),
        "PPO",
        "default_policy",
        OBSERVATION_SPACE,
        ACTION_SPACE,
    )

    agent_specs = {f"AGENT-{i}": agent_spec for i in range(n_mission)}

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[Path(args.scenario).absolute()],
        agent_specs=agent_specs,
        # set headless to false if u want to use envision
        headless=args.headless,
        visdom=False,
        seed=42,
    )

    agents = {_id: agent_spec.build_agent() for _id, agent_spec in agent_specs.items()}

    for times in range(5):
        observations = env.reset()
        dones = {"__all__": False}
        total_eval_reward = 0
        while not dones["__all__"]:
            agent_actions = {_id: agents[_id].act(obs) for _id, obs in observations.items()}
            observations, rewards, dones, infos = env.step(agent_actions)
            _DEBUG_INFO = True
            if _DEBUG_INFO:
                # import smarts.core.sensors
                x = infos['AGENT-0']['env_obs']
                print(x.events.collisions, x.events.off_road, x.events.off_route, x.events.reached_goal)
                print(x.ego_vehicle_state.speed, x.ego_vehicle_state.lane_id, x.ego_vehicle_state.lane_index)
                print(x.ego_vehicle_state.linear_velocity, x.ego_vehicle_state.angular_velocity)
                for i, y in enumerate(x.neighborhood_vehicle_states):
                    print('Neib ', i, y.lane_id, y.lane_index, y.speed)

                # print(x.GridMapMetadata)
                print(type(x.top_down_rgb.data))
                # plt.imshow(x.top_down_rgb.data)
                # plt.show(block=False)
                # plt.pause(3)
                # plt.close()
                # plt.imshow(x.occupancy_grid_map)
                # plt.show()
                print(type(x.top_down_rgb.data))
                print(x.top_down_rgb.data.dtype)

                # print(x.occupancy_grid_map)
                print(x.occupancy_grid_map.data.shape)
                # ogm_data = x.occupancy_grid_map.data
                # for i in range(ogm_data.shape[0]):
                #     for j in range(ogm_data.shape[1]):
                #         if ogm_data[i][j][0] != 0:
                #             print(i, j, ogm_data[i][j][0])
                # plt.imshow(x.occupancy_grid_map.data)
                # plt.show(block=False)
                # plt.pause(3)
                # plt.close()

                # time.sleep(10)
                # print(infos)
                # total_eval_reward += sum(rewards.values())
                time.sleep(3)

        # log your evaluation score / emit tensorboard metrics
        # print(f"Evaluation: {total_eval_reward:.2f}")

    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
