import tensorflow as tf
import numpy as np
from collections import deque
import random

def exploration(a_dim, d=0.001):
    e = 1
    a = np.arange(a_dim)
    while True:
        if np.random.rand() < e:
            yield np.array([np.random.choice(a)])
        else:
            yield None

        e -= d

        if e <= 0.1:
            d = 0


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

    def __init__(self, state_dim, action_dim, memory=0.99, lr=0.001,
                 tau=0.1, hidden_layers=3, hidden_neurons=32,
                 dropout=0.0, regularization=0.01, scope="dqn", training=True, 
                 max_exp_replay=100000, exp_batch=1024, exp_decay=0.001):

        self.sess  = tf.get_default_session()
        self.lr    = lr
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.memory = memory

        self.exp_replay = ExperienceReplay(max_exp_replay)
        self.exp_batch  = exp_batch

        self.training = training
        self.explorer = exploration(self.a_dim, exp_decay)

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

            self.action      = tf.placeholder(tf.int32,   [None])
            self.expected_qs = tf.placeholder(tf.float32, [None])

            #################################
            # out: [X,..X,...Q,...X,..]     #
            # action: index of Q            #
            # env_util: new Q               #
            #                               #
            # Manipulate action and env to  #
            # C: [X, ..X,...new Q,...X..]   #
            #################################

            A  = tf.reshape(self.expected_qs, [-1, 1])
            B  = tf.one_hot(self.action, self.a_dim)
            A  = tf.multiply(A, B)
            C  = tf.multiply(self.out, B)
            C  = tf.subtract(self.out, C)
            C  = tf.add(C, A)

            self.loss     = tf.losses.mean_squared_error(self.out, C)
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def create_network(self, 
                       layers, 
                       neurons, 
                       dropout,
                       regularization):

        state  = tf.placeholder(tf.float32, [None, self.s_dim])

        regularizer = tf.contrib.layers.l2_regularizer(regularization)
        initializer = tf.contrib.layers.variance_scaling_initializer()
        
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


    def predict(self, state):
        ## TODO FIX BETTER EXPLORATION
        values = self.sess.run(self.out, feed_dict={ self.state: state })

        if self.training:
            suggestion = next(self.explorer)
            if suggestion:
                return suggestion

        return np.argmax(values, axis=1)

    def target_predict(self, state):
        return np.argmax(self.sess.run(self.target_out, feed_dict={ self.target_state: state }))

    def Q_value(self, state):
        return self.sess.run(self.out, feed_dict={self.state: state})

    def target_Q_value(self, state):
        return self.sess.run(self.target_out, feed_dict={self.target_state: state})

    def train(self):
        s1b, a1b, r1b, tb, s2b = self.exp_replay.get(self.exp_batch)

        next_q_pred    = self.target_Q_value(s2b)
        current_q_pred = r1b + self.memory * (1 - tb) * np.amax(next_q_pred, axis=1)

        l = self.sess.run((self.optimize, self.loss),
                                feed_dict={ self.state : s1b
                                          , self.action: a1b
                                          , self.expected_qs: current_q_pred
                                          })[-1]

        self.update_target_network()

        return l

    def update_target_network(self):
        self.sess.run(self.update_op)

    def set_networks_equal(self):
        self.sess.run(self.equal_op)

    def add_experience(self, experience):
        self.exp_replay.add(experience)


