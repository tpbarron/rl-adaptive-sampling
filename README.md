RL Adaptive Sampling

TODO:
  * Implement VPG with and without kalman
  * Implement VPG with 2 layers
  * Implement Linear NPG
  * Implement Linear BRS
  * Implement A2C with parallel threads
  * Implement ACKTR 2 layer

  * Consider improvement bounds
  * LQR case improvement bounds

  * Observe gradient distributions


envCartPole-v0_max_samples50000_batch5000_lr0.05_pioptadam_error0.2_diag1_sos1.0_resetkfx1/0


# A2C 2 layer cartpole
# BRS linear cont cartpole

(py36) [trevor@focus rl]$ python vpg_traj_kf.py --batch-size 5000 --lr 0.01 --max-samples 100000 --env-name CartPole-v0 --pi-optim adam --layers 2 --seed 3 --sos-init 0.0 --kf-error-thresh 0.001
