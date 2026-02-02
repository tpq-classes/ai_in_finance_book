# Financial Q-Learning Agent
#
# (c) Dr. Yves J. Hilpisch
# Artificial Intelligence in Finance
#
# Updated to avoid eager/keras3 issues and remove predict/fit bottlenecks.

import os

# --- IMPORTANT: must be set BEFORE importing tensorflow/keras ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import random
import logging
import numpy as np
from pylab import plt, mpl
from collections import deque

import tensorflow as tf
import tf_keras as keras
from tf_keras.layers import Dense, Dropout
from tf_keras.models import Sequential

plt.style.use("seaborn-v0_8")
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.family"] = "serif"


def set_seeds(seed=100):
    """Function to set seeds for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class TradingBot:
    def __init__(self, hidden_units, learning_rate, learn_env,
                 valid_env=None, val=True, dropout=False):
        self.learn_env = learn_env
        self.valid_env = valid_env
        self.val = val

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99

        self.learning_rate = learning_rate
        self.gamma = 0.5
        self.batch_size = 128

        self.max_treward = 0
        self.averages = list()
        self.trewards = []
        self.performances = list()
        self.aperformances = list()
        self.vperformances = list()

        self.memory = deque(maxlen=2000)

        self.model = self._build_model(hidden_units, learning_rate, dropout)

    def _build_model(self, hu, lr, dropout):
        """Method to create the DNN model."""
        model = Sequential()
        model.add(Dense(
            hu,
            input_shape=(self.learn_env.lags, self.learn_env.n_features),
            activation="relu"
        ))
        if dropout:
            model.add(Dropout(0.3, seed=100))
        model.add(Dense(hu, activation="relu"))
        if dropout:
            model.add(Dropout(0.3, seed=100))
        # Output is (batch, lags, 2) because Dense is applied time-wise
        model.add(Dense(2, activation="linear"))

        model.compile(
            loss="mse",
            optimizer=keras.optimizers.RMSprop(learning_rate=lr),
        )
        return model

    def act(self, state):
        """Take action based on exploration vs exploitation."""
        if random.random() <= self.epsilon:
            return self.learn_env.action_space.sample()

        # Fast inference: avoid model.predict()
        # state expected shape: (1, lags, n_features)
        q = self.model(tf.convert_to_tensor(state, dtype=tf.float32), training=False)

        # Keep the original behavior: use the first time step (index 0)
        # If you intended the latest step, change 0 -> -1.
        q0 = q[0, 0]  # shape (2,)
        return int(tf.argmax(q0).numpy())

    def replay(self):
        """Retrain the DNN model based on batches of memorized experiences."""
        batch = random.sample(self.memory, self.batch_size)

        # states in memory are shape (1, lags, n_features)
        states = np.concatenate([b[0] for b in batch], axis=0).astype(np.float32)        # (B, lags, n_features)
        actions = np.array([b[1] for b in batch], dtype=np.int32)                        # (B,)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)                      # (B,)
        next_states = np.concatenate([b[3] for b in batch], axis=0).astype(np.float32)   # (B, lags, n_features)
        dones = np.array([b[4] for b in batch], dtype=np.bool_)                          # (B,)

        # Q(s, :)
        q_states = self.model(tf.convert_to_tensor(states), training=False).numpy()      # (B, lags, 2)
        # Q(s', :)
        q_next = self.model(tf.convert_to_tensor(next_states), training=False).numpy()  # (B, lags, 2)

        # Original code used [0, 0] indexing; vectorize that:
        q_next_first = q_next[:, 0, :]                                                   # (B, 2)
        max_q_next = np.max(q_next_first, axis=1).astype(np.float32)                      # (B,)

        targets = q_states.copy()
        # target for chosen action at the first time step (index 0)
        # if done: target = reward
        # else:   target = reward + gamma * max_a' Q(s', a')
        updated = rewards + (1.0 - dones.astype(np.float32)) * (self.gamma * max_q_next)
        targets[np.arange(self.batch_size), 0, actions] = updated

        # One training call instead of per-sample fit()
        self.model.train_on_batch(states, targets)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        """Train the DQL agent."""
        for e in range(1, episodes + 1):
            state = self.learn_env.reset()
            state = np.reshape(state, [1, self.learn_env.lags, self.learn_env.n_features])

            for t in range(10000):
                action = self.act(state)
                next_state, reward, done, info = self.learn_env.step(action)
                next_state = np.reshape(next_state, [1, self.learn_env.lags, self.learn_env.n_features])

                self.memory.append([state, action, reward, next_state, done])
                state = next_state

                if done:
                    treward = t + 1
                    self.trewards.append(treward)
                    av = sum(self.trewards[-25:]) / 25
                    perf = self.learn_env.performance

                    self.averages.append(av)
                    self.performances.append(perf)
                    self.aperformances.append(sum(self.performances[-25:]) / 25)
                    self.max_treward = max(self.max_treward, treward)

                    templ = "episode: {:2d}/{} | treward: {:4d} | perf: {:5.3f} | av: {:5.1f} | max: {:4d}"
                    print(templ.format(e, episodes, treward, perf, av, self.max_treward), end="\r")
                    break

            if self.val:
                self.validate(e, episodes)

            if len(self.memory) > self.batch_size:
                self.replay()

        print()

    def validate(self, e, episodes):
        """Validate the performance of the DQL agent."""
        state = self.valid_env.reset()
        state = np.reshape(state, [1, self.valid_env.lags, self.valid_env.n_features])

        for t in range(10000):
            # Fast inference: avoid predict()
            q = self.model(tf.convert_to_tensor(state, dtype=tf.float32), training=False)
            action = int(tf.argmax(q[0, 0]).numpy())

            next_state, reward, done, info = self.valid_env.step(action)
            state = np.reshape(next_state, [1, self.valid_env.lags, self.valid_env.n_features])

            if done:
                treward = t + 1
                perf = self.valid_env.performance
                self.vperformances.append(perf)

                if e % max(1, int(episodes / 6)) == 0:
                    templ = 71 * "="
                    templ += "\nepisode: {:2d}/{} | VALIDATION | treward: {:4d} | perf: {:5.3f} | eps: {:.2f}\n"
                    templ += 71 * "="
                    print(templ.format(e, episodes, treward, perf, self.epsilon))
                break


def plot_treward(agent):
    """Plot the total reward per training episode."""
    plt.figure(figsize=(10, 6))
    x = range(1, len(agent.averages) + 1)
    y = np.polyval(np.polyfit(list(x), agent.averages, deg=3), list(x))
    plt.plot(x, agent.averages, label="moving average")
    plt.plot(x, y, "r--", label="regression")
    plt.xlabel("episodes")
    plt.ylabel("total reward")
    plt.legend()


def plot_performance(agent):
    """Plot the financial gross performance per training episode."""
    plt.figure(figsize=(10, 6))
    x = range(1, len(agent.performances) + 1)
    y = np.polyval(np.polyfit(list(x), agent.performances, deg=3), list(x))
    plt.plot(x, agent.performances[:], label="training")
    plt.plot(x, y, "r--", label="regression (train)")

    if agent.val:
        # validation list might be shorter if validation isn't called each episode
        xv = range(1, len(agent.vperformances) + 1)
        if len(agent.vperformances) >= 4:
            yv = np.polyval(np.polyfit(list(xv), agent.vperformances, deg=3), list(xv))
            plt.plot(xv, agent.vperformances[:], label="validation")
            plt.plot(xv, yv, "r-.", label="regression (valid)")
        else:
            plt.plot(xv, agent.vperformances[:], label="validation")

    plt.xlabel("episodes")
    plt.ylabel("gross performance")
    plt.legend()

