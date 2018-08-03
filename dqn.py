import tensorflow as tf
import numpy as np
from collections import deque
import random


class ExperienceReplay(object):

    def __init__(self, capacity):
        self.buffer   = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, frame):
        self.buffer.append(frame)

    def get(self, batchsz):

        if len(self.buffer) < batchsz:
            batchsz = len(self.buffer)

        choices = random.sample(self.buffer, batchsz)

        sb_1 = []
        ab_1 = []
        rb_1 = []
        db_1 = []
        sb_2 = []

        while batchsz:
            sb1, ab1, rb1, db1, sb2 = choices.pop()

            sb_1.append(sb1)
            ab_1.append(ab1)
            rb_1.append(rb1)
            db_1.append(db1)
            sb_2.append(sb2)

            batchsz -= 1

        """ numpyfy """
        sb_1 = np.array(sb_1)
        ab_1 = np.array(ab_1)
        rb_1 = np.array(rb_1)
        db_1 = np.array(db_1)
        sb_2 = np.array(sb_2)

        return sb_1, ab_1, rb_1, db_1, sb_2


class DQN(object):

    def __init__(self, state_dim, action_dim, memory=0.99, lr=0.01,
                 tau=0.1, hidden_layers=3, hidden_neurons=32,
                 dropout=0.0, regularization=0.01, scope="dqn", training=True, 
                 max_exp_replay=100000, exp_batch=1024, lr_decay=0.001,
                 noise_decay=0.001):

        self.sess  = tf.get_default_session()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.memory = tf.constant(memory, tf.float32)

        self.exp_replay = ExperienceReplay(max_exp_replay)
        self.exp_batch  = exp_batch

        self.training = training

        with tf.variable_scope(scope):
            with tf.variable_scope("pi"):
                self.state, self.out =\
                        self.create_network(hidden_layers,
                                            hidden_neurons,
                                            dropout,
                                            regularization)

            with tf.variable_scope("target_pi"):
                self.target_state, self.target_out =\
                        self.create_network(hidden_layers,
                                            hidden_neurons,
                                            dropout,
                                            regularization)


            pi_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           '{}/pi'.format(scope))

            target_pi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           '{}/target_pi'.format(scope))

            #####################
            # Update Target Ops #
            #####################
            self.update_op = [tpv.assign(tf.multiply(tpv, 1 - tau) +\
                                                   tf.multiply(pv, tau))
                                for tpv, pv in zip(target_pi_vars, pi_vars)]

            self.equal_op = [tpv.assign(pv)
                                for tpv, pv in zip(target_pi_vars, pi_vars)]

            ######################
            # Update Network Ops #
            ######################

            mask          = tf.ones(self.a_dim)
            self.action   = tf.placeholder(tf.int32,   [None])
            self.reward   = tf.placeholder(tf.float32, [None])
            self.terminal = tf.placeholder(tf.int32, [None])

            #################################
            # out: [X,..X,...Q,...X,..]     #
            # action: index of Q            #
            # env_util: new Q               #
            #                               #
            # Manipulate action and env to  #
            # C: [X, ..X,...new Q,...X..]   #
            #################################

            action_mask    = tf.one_hot(self.action, self.a_dim) 
            prev_util_mask = mask - action_mask

            exp_util = self.reward +\
                    tf.to_float(1 - self.terminal) * self.memory * tf.reduce_max(self.target_out, axis=1)

            exp_util = self.out * prev_util_mask + action_mask * tf.expand_dims(exp_util, [1])

            self.loss = tf.losses.huber_loss(self.out, exp_util)

            lr_scale = tf.Variable(1.0, dtype=tf.float32)
            self.optimize = tf.train.AdamOptimizer(learning_rate=lr * lr_scale).minimize(self.loss)


            ##########################
            # Exploration and decays #
            ##########################
            lr_decay    = tf.constant(1 - lr_decay, dtype=tf.float32)
            noise_decay = tf.constant(1 - noise_decay, dtype=tf.float32)
            noise       = tf.Variable(1, dtype=tf.float32)

            batch         = tf.shape(self.out)[0]
            stochastic_a  = tf.random_uniform(dtype=tf.int64, 
                                              minval=0, 
                                              maxval=self.a_dim,
                                              shape=[batch])

            stochastic_c  = tf.random_uniform(dtype=tf.float32,
                                              minval=0,
                                              maxval=1,
                                              shape=[batch]) < noise

            ##############################
            # This is not used.. but meh #
            ##############################
            self.deterministic_target_out = tf.argmax(self.target_out, axis=1)

            self.deterministic_out = tf.argmax(self.out, axis=1)
            self.stochastic_out    = tf.where(stochastic_c, stochastic_a, self.deterministic_out)

            update_lr_op  = lr_scale.assign(lr_scale * lr_decay)
            update_exp_op = noise.assign(noise * noise_decay)

            self.u_exp_lr_op = (update_lr_op, update_exp_op)

            if training:
                self.predict = self._predict
            else:
                self.predict = self._stochastic_predict

    def create_network(self, 
                       layers, 
                       neurons, 
                       dropout,
                       regularization):

        state  = tf.placeholder(tf.float32, [None, self.s_dim])

        regularizer = tf.contrib.layers.l2_regularizer(regularization)
        initializer = tf.contrib.layers.variance_scaling_initializer()
        # initializer = tf.truncated_normal_initializer(mean=0, stddev=0.01)
        
        x = tf.layers.dense(state, 
                            neurons,
                            activation=tf.nn.elu,
                            kernel_initializer=initializer,
                            kernel_regularizer=regularizer)

        x = tf.layers.dropout(x, rate=dropout, training=self.training)

        for _ in range(layers):
            x = tf.layers.dense(x,
                                neurons,
                                activation=tf.nn.elu,
                                kernel_initializer=initializer,
                                kernel_regularizer=regularizer)

            x = tf.layers.dropout(x, rate=dropout, training=self.training)

        out = tf.layers.dense(x, 
                             self.a_dim, 
                             activation=None, 
                             kernel_regularizer=regularizer)

        return state, out

    def _predict(self, state):
        return self.sess.run(self.deterministic_out, feed_dict={ self.state: state })

    def _stochastic_predict(self, state):
        return self.sess.run(self.stochastic_out, feed_dict={ self.state: state })

    def target_predict(self, state):
        return self.sess.run(self.deterministic_target_out, feed_dict={ self.target_state: state })

    def Q_value(self, state):
        return self.sess.run(self.out, feed_dict={self.state: state})

    def target_Q_value(self, state):
        return self.sess.run(self.target_out, feed_dict={self.target_state: state})

    def train(self):
        s1b, a1b, r1b, tb, s2b = self.exp_replay.get(self.exp_batch)

        l = self.sess.run((self.optimize, self.loss),
                                feed_dict={ self.state : s1b
                                          , self.action: a1b
                                          , self.target_state: s2b
                                          , self.reward: r1b
                                          , self.terminal: tb
                                          })[-1]

        self.sess.run(self.u_exp_lr_op)
        self.update_target_network()

        return l

    def update_target_network(self):
        self.sess.run(self.update_op)

    def set_networks_equal(self):
        self.sess.run(self.equal_op)

    def add_experience(self, experience):
        self.exp_replay.add(experience)


