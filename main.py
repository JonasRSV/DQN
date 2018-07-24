import gym
import time
import dqn
import sys
import replay_buffer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


ENV = 'CartPole-v0'

EPOCHS = 200
ACTION_SPACE  = 2
STATE_SPACE   = 4 
LEARNING_RATE = 0.01

FRAME_SZ      = 200
BATCHSZ       = 20
MEMORY        = 0.98

TAU           = 0.1


def train(env, actor, rpbuffer, action_noise=True):
    actor.update_target_network()
    actions = np.arange(ACTION_SPACE)

    EXP = 1
    DEC = 1 - 2e-2

    frames = 0
    for g in range(EPOCHS):
        s1       = env.reset()
        terminal = False

        while not terminal:
            frames += 1
            env.render()

            s = s1.reshape(1, -1)
            action = None
            if action_noise:
                if np.random.rand() > EXP:
                    action = actor.predict(s)[0]
                else:
                    action = np.random.choice(actions)
            else:
                action = actor.predict(s)[0]

            s2, r2, terminal, _ = env.step(action)
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

        if action_noise:
            EXP *= DEC

        print(frames)
        print(EXP)

        print()

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

    env = gym.make(ENV)

    with tf.Session() as sess:
        actor         = dqn.DQN(sess, LEARNING_RATE, ACTION_SPACE, STATE_SPACE, tau=TAU, parameter_noise=True)
        rpbuffer = replay_buffer.ReplayBuffer(FRAME_SZ)

        saver = tf.train.Saver()

        if "-n" in sys.argv:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "model/")
            print("Restored...")

        try: 
            if "-p" in sys.argv:
                print("Playing...")
                play(env, actor, action_noise=False)
            else:
                train(env, actor, rpbuffer)
        except KeyboardInterrupt:
            pass

        saver.save(sess, "model/")


    
