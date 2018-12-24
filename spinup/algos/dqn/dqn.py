import numpy as np
import gym
import time
import torch

from spinup.algos.dqn import core
from spinup.utils.logx import EpochLogger
from spinup.utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from spinup.utils.atari_wrapper import make_atari, wrap_deepmind
from spinup.utils.misc import LinearSchedule
"""

Deep Deterministic Policy Gradient (DDPG)

"""


def dqn(env_fn,
        ac_kwargs=dict(),
        seed=0,
        cuda=True,
        lr=5e-4,
        learning_start=10000,
        train_interval=4,
        target_network_update_freq=1000,
        grad_norm_clip=10,
        dual=True,
        param_noise_std=0.2,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_eps=1e-6,
        steps_per_epoch=5000,
        epochs=200,
        replay_size=int(1e6),
        test_epsilon=0.01,
        save_freq=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        gamma=0.99,
        logger_kwargs=dict(),
        batch_size=64):
    """

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # tf.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    total_steps = steps_per_epoch * epochs

    env, test_env = env_fn(), env_fn()

    if len(env.observation_space.shape) > 2:
        env_id = env.spec.id
        env = wrap_deepmind(make_atari(env_id), frame_stack=True)
        test_env = wrap_deepmind(make_atari(env_id), clip_rewards=False, frame_stack=True)


    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(replay_size, alpha=prioritized_replay_alpha)
        beta_schedule = LinearSchedule(total_steps,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(replay_size)
        beta_schedule = None

    obs_shape = (1, ) + env.observation_space.shape
    agent = core.DQNAgent(obs_shape=obs_shape, action_space=env.action_space, lr=lr, dual=dual, cuda=cuda)

    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_steps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)


    # Count variables
    # var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    # print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not d:
                # Take deterministic actions at test time (noise_scale=0)
                a, q = agent.step(o, test_epsilon)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0


    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        epsilon = exploration.value(t)
        a, q = agent.step(o, epsilon)
        if q is not None:
            logger.store(QVals=q)


        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)

        # Store experience to replay buffer
        replay_buffer.add(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        """
        Perform DDPG updates 
        """
        if t > learning_start and t % train_interval == 0:

            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes = experience

            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None


            inputs = (obses_t, actions, rewards, obses_tp1, dones, weights)
            td_errors, loss = agent.update(inputs, double_q=True, gamma=gamma, grad_norm_clip=grad_norm_clip)

            logger.store(LossQ=loss)
            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)


        if t > learning_start and t % target_network_update_freq == 0:
            # Update target network periodically.
            agent.sync()


        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('QVals', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=1234)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    dqn(lambda : gym.make(args.env),
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)