import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import json

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy
import dreamer4.dataset.dreamer_folder_dataset as dreamer_folder_dataset
from stable_worldmodel.data.format import get_format



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="Generate data from move a cube dataset")
    parser.add_argument('--image_shape', type=int, default=224)
    parser.add_argument('--env_type', type=str, choices=['single', 'double', 'triple', 'quadruple',  'octuple'], default='single')
    parser.add_argument('--episodes', type=int, help="Number of episodes to generate", required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--task_name', type=str, default='move-cube')
    return parser.parse_args()


def generate_data() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "dataset_config.json"), "w") as f:
        json.dump(vars(args), f)


    world = swm.World('swm/OGBCube-v0', num_envs=1, env_type=args.env_type, image_shape=(args.image_shape, args.image_shape),
                  mode='data_collection')
    
    policy = ExpertPolicy()
    world.set_policy(policy)
    

    writer = get_format("dreamer_folder").open_writer(args.output_dir, task_name="move-cube")
    world.collect(writer=writer, episodes=args.episodes, seed=args.output_dir, options={'variation': ['all']})


if __name__ == '__main__':
    generate_data()