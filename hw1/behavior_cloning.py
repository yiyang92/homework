
import numpy as np
import seaborn as sns
import pickle
import subprocess
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import layers
import os
import gym
from sklearn.utils import shuffle
from tensorflow.python import debug as tf_debug

envs = ["Ant-v1", "HalfCheetah-v1", "Hopper-v1", "Humanoid-v1", "Reacher-v1", "Walker2d-v1"]

# 50 rollouts
def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def run_expert():
    script = "./demo.bash"
    process = subprocess.Popen(script.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def load_exp_data(env_name):
    return pickle.load(open('./exp_data/{}.pkl'.format(env_name), 'rb'))

def vis_exp(env_name):
    data = load_exp_data(env_name)['return']

    print(data.shape)
    # total return
    plt.figure()
    sns.tsplot(data=data, color='r', legend='Expert policy', linestyle='--')
    plt.show()

def clone_behavior(env_name):
    # mean, std = np.mean(data['observations'], axis=0), np.std(data['observations'], axis=0) + 1e-6
    data = load_exp_data(env_name)
    obs = data['observations']
    acs = data['actions']
    acs = acs.reshape(-1, acs.shape[-1])

    print(acs[0, :])
    print(np.sum(acs[0, :]))

    # Normalize observation
    mean_, std_ = np.mean(obs, axis=0), np.std(obs, axis=0) + 1e-6
    print(obs.shape)

    o = tf.placeholder(shape=[None, obs.shape[1]], dtype=tf.float32)
    u = tf.placeholder(shape=[None, acs.shape[1]], dtype=tf.float32)

    #variables
    l_one, l_two, l_three = 64, 64, 64
    with tf.variable_scope("vars"):
        w1 = tf.get_variable('w1', [obs.shape[1], l_one], initializer=tf.random_uniform_initializer(-0.05, 0.05))
        b1 = tf.get_variable('b1', [l_one], initializer=tf.zeros_initializer())
        w2 = tf.get_variable('w2', [l_one, l_two], initializer=tf.random_uniform_initializer(-0.05, 0.05))
        b2 = tf.get_variable('b2', [l_two], initializer=tf.zeros_initializer())
        b_out = tf.get_variable('b4', [acs.shape[1]], initializer=tf.zeros_initializer())
        w_out = tf.get_variable('out', [l_two, acs.shape[1]], initializer=tf.random_uniform_initializer(-0.05, 0.05))
        mean = tf.get_variable('obs_mean', [obs.shape[1]], initializer=tf.constant_initializer(mean_))
        std = tf.get_variable('obs_std', [obs.shape[1]], initializer=tf.constant_initializer(std_))
    # Normalization
    o_norm = tf.divide(tf.subtract(o, mean), std)
    # Model
    a1 = tf.nn.tanh(tf.matmul(o_norm, w1)+b1)
    a2 = tf.nn.tanh(tf.matmul(a1, w2)+b2)
    out = tf.matmul(a2, w_out)+b_out

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
            for s in range(obs.shape[0]//batch_size):
                obs_batch = obs[s * batch_size: (s + 1) * batch_size, :]
                acs_batch = acs[s * batch_size: (s + 1) * batch_size, :]
                _, loss_ = sess.run([optimizer, loss], feed_dict={o: obs_batch, u: acs_batch})
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(u,1), tf.arg_max(out,1)), tf.float32))
            print("Epoch number: {}, loss {}, accuracy: {}".format(ee, loss_, accuracy.eval(feed_dict={o:obs, u:acs})))
        return sess, mean, std

def clone_behavior_k(env_name):
    from keras.models import Sequential
    from keras.layers import Dense, Lambda
    from keras.optimizers import Adam
    import keras

    data = load_exp_data(env_name)
    obs = data['observations']
    acs = data['actions']
    acs = acs.reshape(-1, acs.shape[-1])

    mean, std = np.mean(data['observations'], axis=0), np.std(data['observations'], axis=0) + 1e-6

    model = Sequential([
        Lambda(lambda x: (x - mean) / std, batch_input_shape=(None, obs.shape[1])),
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(acs.shape[1])
    ])

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=keras.losses.mean_squared_error, metrics=['mse', 'accuracy'])
    x, y = shuffle(data['observations'], data['actions'].reshape(-1, acs.shape[-1]))
    # x= (x - mean) / std
    model.fit(x, y,
              validation_split=0.1,
              batch_size=256,
              nb_epoch=30,
              verbose=2)
    return model


def test_run_k(env, model, roll_outs):
    env = gym.make(env)

    returns = []
    observations = []
    actions = []
    num_steps = []

    # with tf.Session() as sess:
    for i in range(roll_outs):
        print('iter', i)
        obs = env.reset()
        max_steps = env.spec.timestep_limit

        done = False
        totalr = 0.
        steps = 0

        while not done:
            action = model.predict(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
        num_steps.append(steps)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))


def test_run(env, roll_outs, sess):
    env = gym.make(env)

    returns = []
    observations = []
    actions = []
    num_steps = []

    # with tf.Session() as sess:
    for i in range(roll_outs):
        print('iter', i)
        obs = env.reset()
        acs_shape = env.action_space.shape[0]
        max_steps = env.spec.timestep_limit

        done = False
        totalr = 0.
        steps = 0
        def policy_func(obs):
            o = tf.constant(obs, dtype=tf.float32)
            # mean, std = np.mean(obs), np.std(obs) + 1e-6
            # o = (o-mean)/std
            with tf.variable_scope("vars", reuse=True) as scope:
                ww1 = tf.get_variable('w1')
                ww2 = tf.get_variable('w2')
                bb1 = tf.get_variable('b1')
                bb2 = tf.get_variable('b2')
                bb_out = tf.get_variable('b4')
                ww_out = tf.get_variable('out')
                mean = tf.get_variable('obs_mean')
                std = tf.get_variable('obs_std')

            # Make some FF network
            o_norm = tf.divide(tf.subtract(o, mean), std)
            a11 = tf.tanh(tf.matmul(o_norm, ww1) + bb1)
            a22 = tf.tanh(tf.matmul(a11, ww2) + bb2)
            out = tf.matmul(a22, ww_out) + bb_out
            # sess.run(tf.global_variables_initializer())
            return sess.run(out)


        while not done:
            action = policy_func(obs[None, :])
            # print(action)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            # env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
        num_steps.append(steps)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    return {'observations': np.array(observations),
            'actions': np.array(actions),
            'returns': np.array(returns),
            'steps': np.array(num_steps)
            }

def print_info():
    for n in range(len(envs)):
        data = load_exp_data(envs[n])
        obs = data['observations']
        acs = data['actions']
        acs = acs.reshape(-1, acs.shape[-1])

        print("Env name: {} Obs shape: {}, act shape: {}".format(envs[n], obs.shape, acs.shape))

if __name__ == '__main__':
    # run_expert()
    # vis_exp(envs[2] + '.pkl')
    # print_info()
    # model = clone_behavior_k(envs[3])
    # test_run_k(envs[3], model, 1)
    sess, mean, std = clone_behavior(envs[0])
    cloning_data = test_run(envs[0], 5, sess)
