from keras.models import Model, Sequential
from keras.layers import *
import numpy as np
import tensorflow as tf
import gym
import random
from enum import Enum
import csv

# CONSTANTS ---
env = gym.make('BreakoutDeterministic-v0')
test_env = gym.make('BreakoutDeterministic-v4')
STATE_SHAPE = env.observation_space.shape
ACTION_SIZE = env.action_space.n

NUM_TRAIN_STEPS = 50000000
TEST_VISUALIZE = False
TEST_FREQ = 50
TEST_EP_NUM = 5

NOOP_MAX = 30
WARMUP = 50000
MEM_CAPACITY = 750000

SCREEN_DIMS = (84, 84)
SCREEN_STACK = 4  # Currently only works with 4 !!! (Don't change)
NEW_STATE_SHAPE = SCREEN_DIMS + (SCREEN_STACK,)

INIT_EPS = 1.0
FINAL_EPS = 0.1
FINAL_EXP_FRAME = 1000000
TEST_EPS = 0.05

UPDATE_FREQ = 4
MINIBATCH_SIZE = 32
GAMMA = 0.99
TAR_UPDATE_WAIT = 10000
LEARNING_RATE = 0.00025
HUBER_LOSS = True
OPTIMIZER = "Graves_RMPSProp"  # Can also be "RMPSProp" or "Adam"
# -------------

class RunType(Enum):
    RAND_FILL = 1
    TRAIN = 2
    TEST = 3

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def add(self, sample):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, amount):
        return random.sample(self.memory, amount)

    def is_full(self):
        if len(self.memory) == self.capacity:
            return True
        return False

class StateProcessor:
    def __init__(self):
        self.create_graph()

    # Using TF for some operations so need to make a graph to run a session
    def create_graph(self):
        self.input_state = tf.placeholder(tf.uint8, shape=STATE_SHAPE, name='state_proc_input_state')
        self.proc_state = tf.image.rgb_to_grayscale(self.input_state)
        # self.proc_state = tf.image.crop_to_bounding_box(self.proc_state, 34, 0, 160, 160)
        self.proc_state = tf.image.resize_images(self.proc_state, [SCREEN_DIMS[0], SCREEN_DIMS[1]], method=tf.image.ResizeMethod.BILINEAR)
        self.proc_state = tf.cast(self.proc_state, tf.uint8)
        self.proc_state = tf.squeeze(self.proc_state)

    # Used when a new state (or initial state) is spit out by the env.
    def process_state(self, next_state, sess, current_state, done):
        if done:
            return_state = np.zeros(shape=SCREEN_DIMS, dtype=np.uint8)
        else:
            if next_state.shape != SCREEN_DIMS:
                return_state = sess.run(self.proc_state, {self.input_state: next_state})
            else:
                return_state = next_state
        if current_state is None:
            assert not done
            return_state = [return_state, return_state, return_state, return_state]
        else:
            return_state = [current_state[1], current_state[2], current_state[3], return_state]
        return return_state


class DQN:
    def __init__(self):
        self.state_processor = StateProcessor()
        self.memory = ReplayMemory(MEM_CAPACITY)
        self.eps = INIT_EPS

        # Compiling -------------------
        self.beh_model = self.create_model()
        self.tar_model = self.create_model()

        self.sess = tf.Session()
        K.set_session(self.sess)
        K.manual_variable_initialization(True)
        self.create_placeholders()
        self.create_beh_graph()
        self.create_tar_graph()
        self.sess.run(tf.global_variables_initializer())
        # -----------------------------

    # COMPILING --------------------------------------------------------------------------------------------------------
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=NEW_STATE_SHAPE))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(ACTION_SIZE))
        return model

    def create_placeholders(self):
        state_shape = (None,) + NEW_STATE_SHAPE
        self.states = tf.placeholder(tf.float32, shape=state_shape, name='states')
        self.actions = tf.placeholder(tf.uint8, shape=(None,), name='actions')
        self.targets = tf.placeholder(tf.float32, shape=(None,), name='targets')

    def create_beh_graph(self):
        states = self.states / 255.0

        self.beh_output = self.beh_model(states)
        outputs_vec = tf.reduce_sum(tf.multiply(tf.one_hot(self.actions, ACTION_SIZE), self.beh_output), axis=1)

        if HUBER_LOSS:
            loss = tf.losses.huber_loss(self.targets, outputs_vec)
        else:
            errors = tf.subtract(self.targets, outputs_vec)
            loss = tf.square(errors)
            loss = tf.reduce_mean(loss)

        if OPTIMIZER is "RMSProp":
            self.minimize = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=0.0, epsilon=0.01, decay=0.95, centered=True).minimize(loss)
        elif OPTIMIZER is "Graves_RMPSProp":
            self.minimize = self.graves_rmsprop_optimizer(loss, LEARNING_RATE, 0.95, 0.01, 0.0)
        else:  # Adam
            self.minimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # Credit for python impl. of graves rmsprop: https://github.com/cgel
    def graves_rmsprop_optimizer(self, loss, learning_rate, rmsprop_decay, rmsprop_constant, gradient_clip):
        with tf.name_scope('rmsprop'):
            optimizer = None
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(loss)

            grads = []
            params = []
            for p in grads_and_vars:
                if p[0] == None:
                    continue
                grads.append(p[0])
                params.append(p[1])
            # grads = [gv[0] for gv in grads_and_vars]
            # params = [gv[1] for gv in grads_and_vars]
            if gradient_clip > 0:
                grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

            square_grads = [tf.square(grad) for grad in grads]

            avg_grads = [tf.Variable(tf.zeros(var.get_shape()))
                         for var in params]
            avg_square_grads = [tf.Variable(
                tf.zeros(var.get_shape())) for var in params]

            update_avg_grads = [
                grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + tf.scalar_mul((1 - rmsprop_decay), grad_pair[1]))
                for grad_pair in zip(avg_grads, grads)]
            update_avg_square_grads = [
                grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1])))
                for grad_pair in zip(avg_square_grads, grads)]
            avg_grad_updates = update_avg_grads + update_avg_square_grads

            rms = [tf.sqrt(avg_grad_pair[1] - tf.square(avg_grad_pair[0]) + rmsprop_constant)
                   for avg_grad_pair in zip(avg_grads, avg_square_grads)]

            rms_updates = [grad_rms_pair[0] / grad_rms_pair[1]
                           for grad_rms_pair in zip(grads, rms)]
            train = optimizer.apply_gradients(zip(rms_updates, params))

            return tf.group(train, tf.group(*avg_grad_updates))

    def create_tar_graph(self):
        states = self.states / 255.0

        self.tar_output = self.tar_model(states)

    # ------------------------------------------------------------------------------------------------------------------
    def update_implicit_policy(self, step):
        if step <= FINAL_EXP_FRAME:
            self.eps = FINAL_EPS + (INIT_EPS - FINAL_EPS) * np.maximum(0, (FINAL_EXP_FRAME - step)) / FINAL_EXP_FRAME

    def act(self, state, run_type):
        if run_type is RunType.RAND_FILL:
            return np.random.random_integers(0, ACTION_SIZE - 1)

        if run_type is RunType.TRAIN:
            if np.random.uniform() < self.eps:
                return np.random.random_integers(0, ACTION_SIZE - 1)
            return np.argmax(self.get_q_vals(state))

        if run_type is RunType.TEST:
            if np.random.uniform() < TEST_EPS:
                return np.random.random_integers(0, ACTION_SIZE - 1)
            return np.argmax(self.get_q_vals(state))

    def get_q_vals(self, state):
        state_stack = np.stack(state, axis=2)
        state_stack = np.expand_dims(state_stack, axis=0)
        output = self.predict(state_stack, target=False).flatten()
        return output

    def remember(self, state, action, reward, next_state, done):
        sample = (state, action, reward, next_state, done)
        self.memory.add(sample)

    def predict(self, states, target):
        if not target:
            return self.sess.run(self.beh_output, {self.states: states})
        else:
            return self.sess.run(self.tar_output, {self.states: states})

    def return_inputs_targets_errors(self, minibatch):
        shape_s = (MINIBATCH_SIZE,) + NEW_STATE_SHAPE
        states = np.zeros(shape=shape_s)
        next_states = np.zeros(shape=shape_s)
        i = 0
        for sample in minibatch:
            states[i] = np.stack(sample[0], axis=2)
            next_states[i] = np.stack(sample[3], axis=2)
            i += 1

        tar_predictions = self.predict(next_states, target=True)

        targets = np.zeros(MINIBATCH_SIZE)
        actions = np.zeros(MINIBATCH_SIZE)

        index = 0
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = reward + GAMMA * np.amax(tar_predictions[index])
            targets[index] = target
            actions[index] = action
            index += 1

        return (states, targets, actions)

    def update_params(self, step):
        minibatch_data = self.memory.sample_batch(MINIBATCH_SIZE)
        (x, y, actions) = self.return_inputs_targets_errors(minibatch_data)
        feed_dict = {self.states: x, self.targets: y, self.actions: actions}
        self.sess.run(self.minimize, feed_dict)

    def check_update_target_model(self, step):
        if step % TAR_UPDATE_WAIT == 0:
            self.tar_model.set_weights(self.beh_model.get_weights())


class Runner:
    def __init__(self, run_type, agent, nb_steps, nb_ep, file_path, current_train_step):

        if run_type is RunType.TRAIN or run_type is RunType.RAND_FILL:
            self.env = env
        else:
            self.env = test_env

        self.run_type = run_type
        self.agent = agent
        self.nb_steps = nb_steps
        self.nb_ep = nb_ep
        self.file_path = file_path
        self.current_train_step = current_train_step

        self.current_avg = 0
        self.current_max = -1
        self.current_temp_avg = 0

        if self.run_type is RunType.TRAIN:
            f = open(self.file_path + '.csv', "w")
            wr = csv.writer(f)
            wr.writerow(["Episode", "Step", "Total Rew", "Current Avg Rew", "Current Max Rew"])

    def run(self):
        # Fill memory (warmup)
        if self.run_type is RunType.TRAIN:
            mem_filler = Runner(RunType.RAND_FILL, self.agent, None, None, None, current_train_step=None)
            mem_filler.run()

        self.print_flag()

        current_total_step = 0
        current_episode = 1  # MUST START AT 1
        current_ep_step, state, noop = self.reset_episode(None)
        total_ep_reward = 0

        if self.run_type is not RunType.RAND_FILL:
            print("Episode {}...".format(current_episode))

        while self.check_loop(current_total_step, current_episode):

            if self.run_type is RunType.TRAIN and current_ep_step < noop:
                action = 0
            else:
                action = self.agent.act(state, self.run_type)

            if self.run_type is RunType.TEST and TEST_VISUALIZE:
                self.env.render()

            # This makes it so the episode ends if a life is lost but doesnt reset the screen unless the lives == 0;
            # this is what the original DQN paper does
            reset_env = False
            lives_before = self.env.unwrapped.ale.lives()
            next_state, reward, done, _ = self.env.step(action)
            lives_after = self.env.unwrapped.ale.lives()
            if lives_after == 0:
                reset_env = True
            if lives_before > lives_after:
                done = True

            next_state = self.agent.state_processor.process_state(next_state, self.agent.sess, state, done)

            total_ep_reward += reward  # Just for stats
            reward = np.clip(reward, -1.0, 1.0)

            self.remember(state, action, reward, next_state, done)

            self.update_models_and_policy(current_total_step)

            if done:
                if self.run_type is not RunType.RAND_FILL:
                    self.current_max = np.maximum(self.current_max, total_ep_reward)
                    self.current_avg = self.current_avg + ((total_ep_reward - self.current_avg) / current_episode)
                    self.print_and_save_to_csv(current_episode, current_total_step, total_ep_reward, self.current_avg, self.current_max)
                    # self.agent.save_model(current_total_step)

                # TEST
                if current_episode % TEST_FREQ == 0 and self.run_type is RunType.TRAIN:
                    Runner(RunType.TEST, self.agent, None, 5, "breakout_test", current_train_step=current_total_step).run()

                if reset_env:
                    current_ep_step, state, noop = self.reset_episode(None)
                else:
                    current_ep_step, state, noop = self.reset_episode(next_state[3])

                total_ep_reward = 0
                current_episode += 1

                if self.run_type is not RunType.RAND_FILL:
                    print("Episode {}...".format(current_episode))
            else:
                state = next_state
                current_ep_step += 1

            current_total_step += 1

        print("...Done")

    def print_flag(self):
        if self.run_type is RunType.RAND_FILL:
            print("Randomly filling agent memory...")
        elif self.run_type is RunType.TRAIN:
            print("Training agent...")
        elif self.run_type is RunType.TEST:
            print("Testing agent...")

    def reset_episode(self, next_state):
        noop = np.random.randint(0, NOOP_MAX)
        if next_state is None:
            reset_state = self.env.reset()
        else:
            reset_state = next_state
        return 0, self.agent.state_processor.process_state(reset_state, self.agent.sess, None, False), noop

    def check_loop(self, current_total_step, current_ep):
        if self.run_type is RunType.RAND_FILL:
            if self.agent.memory.is_full() or current_total_step == WARMUP:
                return False
            else:
                return True
        if self.run_type is RunType.TRAIN:
            if current_total_step == self.nb_steps:
                return False
            else:
                return True
        if self.run_type is RunType.TEST:
            if current_ep == (TEST_EP_NUM + 1):
                return False
            else:
                return True

    def remember(self, state, action, reward, next_state, done):
        if self.run_type is RunType.TEST:
            return
        self.agent.remember(state, action, reward, next_state, done)

    def update_models_and_policy(self, current_total_step):
        if self.run_type is RunType.RAND_FILL or self.run_type is RunType.TEST:
            return
        if current_total_step % UPDATE_FREQ == 0:
            self.agent.update_params(current_total_step)
        self.agent.update_implicit_policy(current_total_step)
        self.agent.check_update_target_model(current_total_step)

    def print_and_save_to_csv(self, current_ep, current_step, rew, avg_rew, max_rew):
        if self.run_type is RunType.TRAIN:
            print("Step: {}".format(current_step))
            print("Eps: {}".format(self.agent.eps))
            print("Rew: {}".format(rew))
            print("Avg Rew: {}".format(avg_rew))
            print("Max Rew: {}\n".format(max_rew))
            row = [current_ep, current_step, rew, avg_rew, max_rew]
        else:
            print("Step: {}".format(current_step))
            print("Rew: {}".format(rew))
            print("Avg Rew: {}".format(avg_rew))
            row = [self.current_train_step, avg_rew]

        if self.run_type is RunType.TRAIN or (self.run_type is RunType.TEST and current_ep == TEST_EP_NUM):
            with open(self.file_path + '.csv', 'a', newline='') as csvfile:
                wr = csv.writer(csvfile)
                wr.writerow(row)


def main():
    agent = DQN()
    # Trains and tests agent periodically
    # MAKE SURE TO CLEAR THE TEST CSV FILE AFTER RESTARTING, if using same test CSV as before AND...
    # make sire to make the test csv file before hand for now
    train = Runner(RunType.TRAIN, agent, NUM_TRAIN_STEPS, None, "breakout_train", current_train_step=None)
    train.run()

if __name__ == "__main__":
    main()
