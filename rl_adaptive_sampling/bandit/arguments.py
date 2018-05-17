import argparse

parser = argparse.ArgumentParser(description='RL Pol Grad')
parser.add_argument('--lr', type=float, default=0.025,
                    help='learning rate (default: 0.025)')
parser.add_argument('--kf-error-thresh', type=float, default=0.025,
                    help='kf error update thresh (default: 0.025)')
parser.add_argument('--nu', type=float, default=0.1,
                    help='scaling parameter for noise in BRS (default: 0.1)')
parser.add_argument('--sos-init', type=float, default=0.0,
                    help='sos prior for kalman noise approx (default: 0.0)')
parser.add_argument('--func', type=str, default='parabola',
                    help='parabola, ndquad, quartic, rosen (default: parabola)')
parser.add_argument('--log-dir', type=str, default='/tmp/rl_kalman/',
                    help='dir to save logs (default: /tmp/rl_kalman/)')
parser.add_argument('--max-samples', type=int, default=10000,
                    help='total samples (default: 10000)')
parser.add_argument('--batch-size', type=int, default=10000,
                    help='batch size if no kalman (default: 10000)')
parser.add_argument('--min-std', type=float, default=0.1,
                    help='minimum std of sampling dist (default: 0.1)')
parser.add_argument('--no-kalman', action="store_true", default=False,
                    help='dont use kalman (default: False)')
parser.add_argument('--noisy-objective', action="store_true", default=False,
                    help='add noise to objective (default: False)')
parser.add_argument('--use-diagonal-approx', action="store_true", default=False,
                    help='use diagonal approximation in Kalman filter (default: False)')
parser.add_argument('--use-last-error', action="store_true", default=False,
                    help='use last step empirical error to set the current initial \
                    error in Kalman filter (default: False)')
parser.add_argument('--reset-kf-state', action='store_true', default=False,
                    help='reset kf state to 0 at each iter (default: false)')
parser.add_argument('--reset-obs-noise', action='store_true', default=False,
                    help='reset obs noise in kf at each iter (default: false)')
parser.add_argument('--kf-window-size', type=int, default=100,
                    help='variance window size (default: 100)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
args = parser.parse_args()

def get_args():
    return args
