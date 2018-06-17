import tensorflow as tf
import numpy as np



def trainable_weigth(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, dtype=tf.float32)


class DQN(object):

    def __init__(self, session, learning_rate, action_space, state_space):
        self.session       = session
        self.learning_rate = learning_rate
        self.action_space  = action_space
        self.state_space   = state_space

        state, out = self.create_network()
        action, label, loss, optimize = self.create_training(out)


        """ Tensors needed for DQN. """
        self.state    = state
        self.out      = out
        self.action   = action
        self.label    = label
        self.loss     = loss
        self.optimize = optimize

        self.choice   = tf.argmax(out, axis=1)


    def create_network(self):

        """ Trainable weights. """
        state    = tf.placeholder(tf.float32, [None, self.state_space])

        dense0_w = trainable_weigth([self.state_space, 64])
        dense0_b = trainable_weigth([64])

        dense1_w = trainable_weigth([64, 64])
        dense1_b = trainable_weigth([64])

        out_w    = trainable_weigth([64, self.action_space])
        out_b    = trainable_weigth([self.action_space])

        """ Graph """

        h0       = tf.matmul(state, dense0_w)
        h0       = tf.nn.tanh(h0 + dense0_b)

        h1       = tf.matmul(h0, dense1_w)
        h1       = tf.nn.tanh(h1 + dense0_b)

        out      = tf.matmul(h1, out_w)
        out      = tf.nn.relu(out + out_b)

        return state, out

    def create_training(self, out):

        action    = tf.placeholder(tf.int32,   [None])
        label     = tf.placeholder(tf.float32, [None])

        lshape    = tf.reshape(label, [-1, 1])
        choice    = tf.one_hot(action, self.action_space)
        choice    = tf.multiply(choice, lshape)

        loss      = tf.losses.mean_squared_error(out, choice)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return action, label, loss, optimizer

    def predict(self, state):
        return self.session.run(self.choice, feed_dict={ self.state: state })


    def Q_value(self, state):
        return self.session.run(self.out,
                                feed_dict={self.state: state})

    def train(self, state, action, label):
        _, l = self.session.run((self.optimize, self.loss),
                                feed_dict={ self.state : state
                                          , self.action: action
                                          , self.label : label
                                          })
        return l

