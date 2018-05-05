import argparse

parser = argparse.ArgumentParser(description='RL Pol Grad')
parser.add_argument('--lr', type=float, default=0.025,
                    help='learning rate (default: 0.025)')
parser.add_argument('--kf-error-thresh', type=float, default=0.025,
                    help='kf error update thresh (default: 0.025)')
parser.add_argument('--log-dir', type=str, default='/tmp/rl_kalman/',
                    help='dir to save logs (default: /tmp/rl_kalman/)')
parser.add_argument('--n-iters', type=int, default=100,
                    help='number of optimization steps (default: 100)')
parser.add_argument('--batch-size', type=int, default=10000,
                    help='batch size if no kalman (default: 10000)')
parser.add_argument('--min-std', type=float, default=0.1,
                    help='minimum std of sampling dist (default: 0.1)')
parser.add_argument('--no-kalman', action="store_true", default=False,
                    help='dont use kalman (default: False)')
parser.add_argument('--noisy-objective', action="store_true", default=False,
                    help='add noise to objective (default: False)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
args = parser.parse_args()

def get_args():
    return args
