import argparse
from space_invaders import SpaceInvader

# Hyperparameters
NUM_FRAME = 1000000

parser = argparse.ArgumentParser(description="Train and test DQN on Space Invaders")

# Parse arguments
parser.add_argument("-m", "--mode", type=str, action='store', required=True)
parser.add_argument("-l", "--load", type=str, action='store', required=False)
parser.add_argument("-x", "--statistics", action='store_true', required=False)
parser.add_argument("-v", "--view", action='store_true', required=False)

args = parser.parse_args()
print(args)

game_instance = SpaceInvader()

if args.load:
    game_instance.load_network(args.load)

if args.mode == "train":
    game_instance.train(NUM_FRAME)

if args.statistics:
    stat = game_instance.calculate_mean()
    print("Game Scores : ")
    print(stat)

if args.view:
    game_instance.simulate()

