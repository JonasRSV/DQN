import tensorflow as tf
import numpy as np



def trainable_weigth(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, dtype=tf.float32)

HIDDEN_CONNECTIONS = 64

class DQN(object):

    def __init__(self, session, learning_rate, action_space, state_space, var_index=0, tau=0.1):
        self.session       = session
        self.learning_rate = learning_rate
        self.action_space  = action_space
        self.state_space   = state_space

        state, out = self.create_network("vanilla")
        action, label, loss, optimize = self.create_training(out, "vanilla")

        vanilla_variables = tf.trainable_variables()[var_index:]

        target_state, target_out = self.create_network("target")
        target_action, target_label, _, _ = self.create_training(target_out, "target")

        target_variables = tf.trainable_variables()[var_index + len(vanilla_variables):]

        update_op = [target_var.assign(tf.multiply(target_var, 1 - tau) +\
                                       tf.multiply(vanilla_var, tau))
                            for target_var, vanilla_var in zip(target_variables, vanilla_variables)]

        ####################################################
        # Good explanation of Double DQN vs Target Network #
        # This implementation can easily use any of them.  #
        ####################################################

        # https://datascience.stackexchange.com/questions/32246/q-learning-target-network-vs-double-dqn


        """ Tensors needed for DQN. """
        self.state    = state
        self.out      = out
        self.action   = action
        self.label    = label

        self.target_state  = target_state
        self.target_out    = target_out
        self.target_action = target_action
        self.target_label  = target_label

        self.choice        = tf.argmax(out, axis=1)
        self.target_choice = tf.argmax(target_out, axis=1)

        self.loss     = loss
        self.optimize = optimize

        self.update_op = update_op

    def create_network(self, name):

        with tf.name_scope(name):
            """ Trainable weights. """
            state    = tf.placeholder(tf.float32, [None, self.state_space])

            dense0_w = trainable_weigth([self.state_space, HIDDEN_CONNECTIONS])
            dense0_b = trainable_weigth([HIDDEN_CONNECTIONS])

            dense1_w = trainable_weigth([HIDDEN_CONNECTIONS, HIDDEN_CONNECTIONS])
            dense1_b = trainable_weigth([HIDDEN_CONNECTIONS])

            out_w    = trainable_weigth([HIDDEN_CONNECTIONS, self.action_space])
            out_b    = trainable_weigth([self.action_space])

            """ Graph """

            h0       = tf.matmul(state, dense0_w)
            h0       = tf.nn.tanh(h0 + dense0_b)

            h1       = tf.matmul(h0, dense1_w)
            h1       = tf.nn.tanh(h1 + dense0_b)

            out      = tf.matmul(h1, out_w)
            out      = out + out_b

        return state, out

    def create_training(self, out, name):

        with tf.name_scope(name):
            action    = tf.placeholder(tf.int32,   [None])
            label     = tf.placeholder(tf.float32, [None])

            label_    = tf.reshape(label, [-1, 1])
            one_hots  = tf.one_hot(action, self.action_space)

            label_    = tf.multiply(label_, one_hots)

            """ Manipulate so that only action Q_value is updated with new Q_value"""
            X         = tf.multiply(out, one_hots)
            X         = tf.subtract(out, X)

            label_    = tf.add(X, label_)


            loss      = tf.losses.mean_squared_error(out, label_)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss)

        return action, label, loss, optimizer

    def predict(self, state):
        return self.session.run(self.choice, feed_dict={ self.state: state })

    def target_predict(self, state):
        return self.session.run(self.target_choice, feed_dict={ self.target_state: state })

    def Q_value(self, state):
        return self.session.run(self.out, feed_dict={self.state: state})

    def target_Q_value(self, state):
        return self.session.run(self.target_out, feed_dict={self.target_state: state})

    def train(self, state, action, label):
        _, l = self.session.run((self.optimize, self.loss),
                                feed_dict={ self.state : state
                                          , self.action: action
                                          , self.label : label
                                          })
        return l

    def update_target_network(self):
        self.session.run(self.update_op)

