import gym
import time
import dqn
import sys
import replay_buffer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

GENERATIONS = 10000

ACTION_SPACE  = 2
STATE_SPACE   = 4 
LEARNING_RATE = 0.01

FRAME_SZ      = 200
BATCHSZ       = 32
MEMORY        = 0.95
DECAY         = 1 - 2e-2
EXPLORATION   = 1


def train(env, actor, rpbuffer):
    global GENERATIONS, EXPLORATION, DECAY

    actor.update_target_network()

    plt.style.use('dark_background')
    actions = np.arange(ACTION_SPACE)

    generations   = []
    rewards       = []
    for g in range(GENERATIONS):
        s1       = env.reset()
        terminal = False

        reward = 0
        while not terminal:
            env.render()

            s = s1.reshape(1, -1)

            action = None
            if np.random.rand() > EXPLORATION:
                action = actor.predict(s)[0]
            else:
                action = np.random.choice(actions)

            s2, r2, terminal, _ = env.step(action)
            reward += r2

            rpbuffer.add((s1, action, r2, terminal, s2))
            s1 = s2

            s1b, a1b, r1b, dd, s2b = rpbuffer.get(BATCHSZ)
            Qvalues = actor.target_Q_value(s2b)

            yi = []
            for d, r, Q in zip(dd, r1b, Qvalues):
                if d:
                    yi.append(r)
                else:
                    yi.append(r + MEMORY * max(Q))

            l = actor.train(s1b, a1b, yi)

            actor.update_target_network()

        EXPLORATION = EXPLORATION * DECAY

        generations.append(g)
        rewards.append(reward)
        
        sys.stdout.write("\rEXPLORATION {} \r".format(EXPLORATION))
        plt.plot(generations, rewards)
        plt.pause(0.001)

    env.close()


def play(env, actor, games=20):
    for i in range(games):
        terminal = False
        s0 = env.reset()


        while not terminal:
            env.render()
            s0 = s0.reshape(1, -1)
            action = actor.predict(s0)[0]

            s0, _, terminal, _ = env.step(action)

    env.close()


if __name__ == "__main__":

    env = gym.make('CartPole-v0')

    with tf.Session() as sess:
        actor         = dqn.DQN(sess, LEARNING_RATE, ACTION_SPACE, STATE_SPACE)
        rpbuffer = replay_buffer.ReplayBuffer(FRAME_SZ)

        saver = tf.train.Saver()

        if "-r" in sys.argv:
            saver.restore(sess, "model/")
            print("Restored...")
        else:
            sess.run(tf.global_variables_initializer())

        try: 
            if "-p" in sys.argv:
                print("Playing...")
                play(env, actor)
            else:
                train(env, actor, rpbuffer)
        except KeyboardInterrupt:
            pass

        saver.save(sess, "model/")


    
