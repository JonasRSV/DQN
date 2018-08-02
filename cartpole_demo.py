import gym
import dqn
import sys
import numpy as np
import tensorflow as tf
import gym_wrapper


ENV = 'CartPole-v0'

if __name__ == "__main__":

    env = gym.make(ENV)

    with tf.Session() as sess:
        training = None
        if "-n" in sys.argv:
            training = True
        else:
            training = False

        actor = dqn.DQN(4,
                        2,
                        memory=0.99,
                        lr=0.01,
                        tau=0.01,
                        exp_batch=64,
                        training=training)

        saver = tf.train.Saver()
        if "-n" in sys.argv:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "model/cartpole")
            print("Restored...")

        try: 
            if "-p" in sys.argv:
                print("Playing...")
                gym_wrapper.play(env, actor)
            else:
                gym_wrapper.train(env, actor, 50000, render=True)
        except KeyboardInterrupt:
            pass

        saver.save(sess, "model/cartpole")

