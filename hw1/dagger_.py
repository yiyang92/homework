import numpy as np
import seaborn as sns
import pickle
import subprocess
import matplotlib.pyplot as plt
import tensorflow as tf
import load_policy
import tf_util
from tensorflow.contrib import layers
import os
import gym
import tqdm
from sklearn.utils import shuffle
from tensorflow.python import debug as tf_debug


def load_exp_data(env_name):
    return pickle.load(open('./exp_data/{}.pkl'.format(env_name), 'rb'))


def load_policy_fn(env_name):
    print('Gathering expert data')
    print('loading and building expert policy')
    policy_fname = 'experts/{}.pkl'.format(env_name)
    policy_fn = load_policy.load_policy(policy_fname)
    print('loaded and built')
    return policy_fn

def run_expert(env_name, ro, data=None,render=False):
    with tf.Session():
        tf_util.initialize()

        env = gym.make(env_name)
        max_steps = env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        num_steps = []
        policy_fn = load_policy_fn(env_name)
        print("Getting expert data for {} rollouts".format(ro))

        for i in tqdm.tqdm(range(ro)):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = policy_fn(obs[None, :])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps >= max_steps:
                    break
            returns.append(totalr)
            num_steps.append(steps)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'return': np.array(returns),
                       'n_steps': np.array(num_steps)}
        return expert_data


def train_policy(env, data, epochs, ro=10, render=False):
    tf.reset_default_graph()
    obs = data['observations']
    acs = data['actions']
    acs = acs.reshape(-1, acs.shape[-1])

    print(acs[0, :])
    print(np.sum(acs[0, :]))

    # Normalize observation
    mean_, std_ = np.mean(obs, axis=0), np.std(obs, axis=0) + 1e-6

    o = tf.placeholder(shape=[None, obs.shape[1]], dtype=tf.float32)
    u = tf.placeholder(shape=[None, acs.shape[1]], dtype=tf.float32)

    # variables
    l_one, l_two, l_three = 64, 64, 64
    with tf.variable_scope("vars"):
        w1 = tf.get_variable('w1', [obs.shape[1], l_one], initializer=tf.random_uniform_initializer(-0.09, 0.09))
        b1 = tf.get_variable('b1', [l_one], initializer=tf.zeros_initializer())
        w2 = tf.get_variable('w2', [l_one, l_two], initializer=tf.random_uniform_initializer(-0.09, 0.09))
        b2 = tf.get_variable('b2', [l_two], initializer=tf.zeros_initializer())
        b_out = tf.get_variable('b4', [acs.shape[1]], initializer=tf.zeros_initializer())
        w_out = tf.get_variable('out', [l_two, acs.shape[1]], initializer=tf.random_uniform_initializer(-0.09, 0.09))
        mean = tf.get_variable('obs_mean', [obs.shape[1]], initializer=tf.constant_initializer(mean_))
        std = tf.get_variable('obs_std', [obs.shape[1]], initializer=tf.constant_initializer(std_))
    # Feature Normalization
    o_norm = tf.divide(tf.subtract(o, mean), std)
    # Model
    a1 = tf.nn.tanh(tf.matmul(o_norm, w1) + b1)
    a2 = tf.nn.tanh(tf.matmul(a1, w2) + b2)
    out = tf.matmul(a2, w_out) + b_out

    # out = tf.Print(out, [tf.shape(out), tf.shape(a2), tf.shape(a1)])
    # define loss function
    lr = 0.001
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr, global_step,
                                               100000, 0.96, staircase=True)

    # loss = tf.losses.mean_squared_error(labels=u, predictions=out)
    loss = tf.reduce_mean(tf.squared_difference(out, u))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # Parameters
    batch_size = 256
    epochs = 30

    # Set CUDA parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        for ee in range(epochs):
            # Shuffle yi xia
            # 1-st variant
            # rng_state = np.random.get_state()
            # np.random.shuffle(obs)
            # np.random.set_state(rng_state)
            # np.random.shuffle(acs)
            # 2-nd variant
            inx = np.random.permutation(len(obs))
            obs, acs = obs[inx], acs[inx]
            for s in range(obs.shape[0] // batch_size):
                obs_batch = obs[s * batch_size: (s + 1) * batch_size, :]
                acs_batch = acs[s * batch_size: (s + 1) * batch_size, :]
                _, loss_ = sess.run([optimizer, loss], feed_dict={o: obs_batch, u: acs_batch})
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(u, 1), tf.arg_max(out, 1)), tf.float32))
            print(
                "Epoch number: {}, loss {}, accuracy: {}".format(ee, loss_, accuracy.eval(feed_dict={o: obs, u: acs})))
        # run it
        observations = []
        returns = []
        env = gym.make(env)
        max_steps = env.spec.timestep_limit
        for r_ in range(ro):
            print('rollout # ', r_)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0

            while not done:
                action = sess.run(out, feed_dict={o: obs[None, :]})
                observations.append(obs)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                # if steps % 100 == 0 and render: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            print("Finished in {} steps".format(steps))
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))

        return np.array(observations), returns

def run_exp_on_ours(env_name, obs, render=False):
    with tf.Session():
        tf_util.initialize()
        actions = []
        policy_fn = load_policy_fn(env_name)
        print("Running expert policy on our observations")

        for ob in tqdm.tqdm(obs):
            action = policy_fn(ob[None, :])
            actions.append(action)
        return actions

def aggregate_datasets(data_or, obs, acs):
    return {"observations": np.concatenate((data_or["observations"], obs), axis=0),
            "actions": np.concatenate((data_or["actions"], acs), axis=0)}

def dagger_main(env_name, iters=20, exp_rol=20, our_rol=30):
    # run expert policy, get D_ex={o_e1, u_e1, o_e2, u_e2, ..., o_en, u_en}
    data = run_expert(env_name, exp_rol)
    rets = []
    ro = our_rol
    rend = False
    for it in range(iters):
        if it%15==0 and it != 0: rend=True
        # step 1, 2 train policy, run it, get D_our={o_1, o_2, ..., o_n}
        obs, returns = train_policy(env_name, data, 20, ro, rend)
        # step 3, run expert policy, get u for D_our
        acs = run_exp_on_ours(env_name, obs)
        # step 4, aggregate two datasets
        data = aggregate_datasets(data, obs, acs)
        # info
        rets.append(returns)
    obs, returns = train_policy(env_name, data, 20, ro, rend)
    rets.append(returns)
    print(rets)
    # Save results
    data = {"observations": data["observations"],
            "actions": data["actions"],
            "returns": np.array(rets)}
    if 'our_dagger' not in os.listdir('.'):
        os.makedirs('./our_dagger')
    pickle.dump(data, open('./our_dagger/{}.pkl'.format(env_name), 'wb'))


def vis_dagger(env_name):
    data = pickle.load(open('./our_dagger/{}.pkl'.format(env_name), 'rb'))
    returns = data["returns"]

    # total return
    plt.figure()
    sns.tsplot(data=returns, color='r', legend='DAgger running results', linestyle='--')
    plt.show()

envs = ["Ant-v1", "HalfCheetah-v1", "Hopper-v1", "Humanoid-v1", "Reacher-v1", "Walker2d-v1"]
if __name__ == '__main__':
    dagger_main(envs[4], 20)
    vis_dagger(env_name=envs[4])