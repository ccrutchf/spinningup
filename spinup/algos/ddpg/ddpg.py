import numpy as np
import gym
import roboschool
import time
from spinup.algos.ddpg import core
from spinup.utils.logx import EpochLogger
from spinup.utils.replay_buffer import Memory
import torch


"""

Deep Deterministic Policy Gradient (DDPG)

"""
def ddpg(env_fn, ac_kwargs=dict(), seed=0, cuda=True, train_interval=100, train_steps=50,
         steps_per_epoch=5000, epochs=200, replay_size=int(1e6), gamma=0.99, hidden_size=64,
         polyak=0.01, pi_lr=1e-4, q_lr=1e-3, batch_size=64, start_steps=1000,
         act_noise=0, param_noise=0.2, max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # tf.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    memory = Memory(limit=replay_size, action_shape=(act_dim, ), observation_shape=(obs_dim, ))

    agent = core.DDPGAgent(obs_dim, act_dim, hidden_size, memory,
                           batch_size=batch_size, tau=polyak, gamma=gamma, action_noise_std=act_noise, cuda=cuda,
                           param_noise_std=param_noise, action_range=(-act_limit, act_limit))




    # Count variables
    # var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    # print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)





    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(agent.step(o, noisy=False)[0])
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            a, q, _, _ = agent.step(o, noisy=True)
            logger.store(QVals=q.mean())
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        agent.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        """
        Perform DDPG updates
        """
        if memory.nb_entries > batch_size and t % train_interval == 0:
            if param_noise > 0:
                distance = agent.adapt_actor_param_noise()

            for _ in range(train_steps):
                # Q-learning update
                value_loss, policy_loss = agent.train()
                logger.store(LossQ=value_loss, LossPi=policy_loss)


        if t > 0 and t % steps_per_epoch == 0:

            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env,}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent(20)

            # Log info about
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
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

    ddpg(lambda : gym.make(args.env),
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
         logger_kwargs=logger_kwargs)