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


# Rebuttle todo:

## Main reviewer concerns:
  * Complexity:
    * Scalability to larger models. Yes, diagonal or block-diagonal approximation is an option for larger models and in this case matrix multiplication and inversion are computationally easy. But more generally, in control tasks there is much research suggesting that (generalized) linear models are sufficiently expressive. We chose to focus on linear models because (1) is it straightforward to ensure the assumptions hold and (2) they are sufficiently expressive for the vast majority of real-world control tasks (THIS IS TOO LARGE OF A CLAIM).
  * Experiments:
    * Compare to additional variance reduction techniques:
      * This is a valid rebuke, though we believe our method should operate in an orthogonal fashion to other approaches and did not incorporate variance reduction techniques with or without KF.
      * Optimal baseline
      * Linear value function baseline.
    * Compare to SGD baseline.
    * Additional tasks:
      * Cartpole - linear.
      * Walker mujoco (RBF)?
    * Testing on non toy problems with a model that breaks assumptions: to test our initial work we wanted to focus on a model and task that would clearly express the inner workings of this method. A larger model, and more complicated tasks make it much more difficult to analyze anything beyond empirical performance.
  * Clarifications:
    * Model based vs model free: This is not model based. It is modeling, with uncertainty, the local loss landspace.
    * Relationship to Adam: we view our method as complementary to an optimizer, like Adam. The optimizers job is to take the gradient at a given iteration (perhaps a history of gradients as well) and perform a parameter update that improves some objective. The KF in our experiments is used to estimate the gradient at each of these iterations, not perform the update as well.
    * The assumptions of the filter rely only on the model. They do not depend on the reward landscape or the task at hand.
    * Present assumptions in terms of RL.
    * What do the matrices of KF represent for RL.

## New experiments
  * Variance reduction techniques:
    * Optimal baseline (incorporate into my code)
    * Linear value fn w/GAE (incorporate into my code)
    * Linear model with RBF.
    * NPG w/ GAE (linear) (use arvind mjrl)
    * TRPO w/ GAE (linear) (use arvind mjrl)
    * Try once with SGD so see if worth it.
  * Tasks:
    * CartPole
    * Walker2D mujoco (RBF)
  * New stats:
    * Wall time.

  * Exps (R: means algo code is ready, RR: means alg is ready and run script is prepared.)
    1) PointMass
      - R: (KF/VPG) Linear policy no baseline SGD
      - R: (KF/VPG) Linear policy no baseline Adam
      - (KF/VPG) Linear policy with linear critic GAE
      - R: VPG (bs 1, 5, 10)
      - NPG with GAE
      - TRPO with GAE
    2) CartPoleContinuous
      - R: (KF/VPG) Linear policy no baseline SGD
      - R: (KF/VPG) Linear policy no baseline Adam
      - (KF/VPG) Linear policy with linear critic GAE
      - R: VPG (bs 1, 5, 10)
      - NPG with GAE
      - TRPO with GAE
    2) Swimmer / Hopper / Walker / Ant
      - Run on best KF variant from CartPole
      - VPG (bs 1, 5, 10)
      - NPG with GAE
      - TRPO with GAE
