import argparse
import sys
import pickle

import numpy as np
from lbd_comp.evaluate_track2 import evaluate_track2


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "-a",
    "--algo",
    type=str,
    default='my_1',
    help="algo",
)

parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default='seed',
    help=20,
)
parser.add_argument(
    "-n",
    "--noisy",
    type=int,
    default=1,
    help="is noisy data",
)

parser.add_argument(
    "-m",
    "--num_traj",
    type=int,
    default=50,
    help="num_traj",
)

parser.add_argument(
    "-o",
    "--output-dir",
    help=(
        "Specifies the output directory to save results. "
        "If no argument is provided, no results will be saved."
    ),
)
parser.add_argument(
    "-f",
    "--force-output",
    action="store_true",
    default=False,
    help="Force overwritting of existing results in the output directory.",
)
parser.add_argument(
    "-d",
    "--debug-output",
    action="store_true",
    default=False,
    help="Generates debug information, such as gif visualizations.",
)
parser.add_argument(
    "-z",
    "--show-viz",
    action="store_true",
    default=False,
    help="Displays visualizations during controller evaluation.",
)
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    default=False,
    help="Enables verbose output.",
)

input_args = sys.argv[1:]
args = parser.parse_args(input_args)
seed = args.seed
algo = args.algo
is_noisy = args.noisy
num_traj = args.num_traj
input_dir = f'data/ROBO/controllers/{algo}/controller/'

traj_dir = 'data/ROBO/target_trajectories/'
if is_noisy==1:
    # noise
    traj_dir = 'output/robo_test_noisy_random_0.5/'

print(f'algo: {algo}, test in noisy data' if is_noisy==1 else f'algo: {algo}, test in noiseless data' )


robo_eval_results = evaluate_track2(
    input_dir,
    traj_dir,
    output_dir=args.output_dir,
    force_output=args.force_output,
    debug_output=args.debug_output,
    show_viz=args.show_viz,
    verbose=args.verbose,
)

rr = []
rr.append(robo_eval_results)
rr = np.array(rr)
np.save(f'results/{algo}_{num_traj}_s_{seed}',rr)
# algo: mlp, rs_linear, 