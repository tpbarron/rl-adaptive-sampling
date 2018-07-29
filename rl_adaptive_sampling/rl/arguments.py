import argparse

parser = argparse.ArgumentParser(description='RL Pol Grad')

parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate (default: 0.0005)')
parser.add_argument('--batch-size', type=int, default=1000,
                    help='training batch size (default: 1000)')
parser.add_argument('--batch-size-traj', type=int, default=10,
                    help='training batch size in trajs (default: 10)')
parser.add_argument('--max-samples', type=int, default=1e6,
                    help='maximum num steps to take (default: 1e6)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.97,
                    help='GAE param (default: 0.97)')
parser.add_argument('--kf-error-thresh', type=float, default=0.0,
                    help='threshold for update expected error (default: 0.0)')
parser.add_argument('--sos-init', type=float, default=0.0,
                    help='sos prior for kalman noise approx (default: 100.0)')
parser.add_argument('--use-diagonal-approx', action="store_true", default=False,
                    help='use diagonal approximation in Kalman filter (default: False)')
# parser.add_argument('--env-name', type=str, default='CartPoleBulletEnv-v0',
#                     help='env to train on (default: CartPoleBulletEnv-v0)')
parser.add_argument('--log-dir', type=str, default='/tmp/rl_kalman/',
                    help='dir to save logs (default: /tmp/rl_kalman/)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--no-kalman', action='store_true', default=False,
                    help='do not use kf estimate (default: false)')
parser.add_argument('--reset-kf-state', action='store_true', default=False,
                    help='reset kf state to 0 at each iter (default: false)')
parser.add_argument('--reset-obs-noise', action='store_true', default=False,
                    help='reset obs noise in kf at each iter (default: false)')

parser.add_argument('--x0', type=float, default=0.5,
                    help='x position (default: 0.5)')
parser.add_argument('--y0', type=float, default=0.5,
                    help='y position (default: 0.0)')
parser.add_argument('--xv0', type=float, default=0.0,
                    help='vel x (default: 0.0)')
parser.add_argument('--yv0', type=float, default=0.0,
                    help='vel y (default: 0.0)')

args = parser.parse_args()

def get_args():
    return args
