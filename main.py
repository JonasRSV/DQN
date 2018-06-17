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
LEARNING_RATE = 0.1

FRAME_SZ      = 200
BATCHSZ       = 32
MEMORY        = 0.8

def train(env, actor, rpbuffer):
    global GENERATIONS

    actions = np.arange(ACTION_SPACE)

    total_frames = []
    generations  = []
    for g in range(GENERATIONS):
        s1     = env.reset()
        dead   = False
        frames = 0
        
        while not dead:
            frames += 1
            env.render()

            
            s = s1.reshape(1, -1)

            action = None
            if np.random.rand() > 0.3:
                action = actor.predict(s)[0]
            else:
                action = np.random.choice(actions)

            s2, r2, dead, _ = env.step(action)
            
            rpbuffer.add((s1, action, r2, dead, s2))

            s1 = s2

            s1b, a1b, r1b, dd, s2b = replay_buffer.get(BATCHSZ)
            Qvalues = actor.Q_value(s2b)

            yi = []
            for d, r, Q in zip(dd, r1b, Qvalues):
                if d:
                    yi.append(r)
                else:
                    yi.append(r + MEMORY * max(Q))

            l = actor.train(s1b, a1b, yi)

        total_frames.append(frames)
        generations.append(g)
        plt.plot(generations, total_frames)

        plt.pause(0.001)



    env.close()

if __name__ == "__main__":

    env = gym.make('CartPole-v0')

    with tf.Session() as sess:
        actor         = dqn.DQN(sess, LEARNING_RATE, ACTION_SPACE, STATE_SPACE)
        replay_buffer = replay_buffer.ReplayBuffer(FRAME_SZ)

        saver = tf.train.Saver()

        if "-r" in sys.argv:
            saver.restore(sess, "model/")
        else:
            sess.run(tf.global_variables_initializer())

        try: 
            train(env, actor, replay_buffer)
        except KeyboardInterrupt:
            pass

        saver.save(sess, "model/")


    
