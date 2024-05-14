import random
import tensorflow as tf

import gymnasium as gym
import gym_puddle
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import os
import json
import optuna

from utils import PQueue
from model import Model, make_env


class DynaQ:
    def __init__(
        self,
        env,
        n,
        alpha,
        gamma,
        epsilon,
        max_steps,
        env_params,
        summary_writer,
        h_weight,
    ):
        self.env = env
        self.grid_size = 40
        self.state_size = [
            int(i)
            for i in (self.env.observation_space.high - self.env.observation_space.low)
            * self.grid_size
        ]

        self.action_size = self.env.action_space.n
        self.q = np.zeros(tuple(self.state_size) + (self.action_size,))
        self.model = Model(
            self.grid_size, self.state_size, self.action_size, env_params, h_weight
        )

        # Priority sampling
        self.p_queue = PQueue()
        # initial thresholds, will change as q values grow/shrink
        # upper threshold is fixed to prevent agent from just updating puddle interactions
        self.p_thresh_lower = 0.0
        self.p_thresh_upper = 40
        self.update_threshold = 0.25
        self.avg_update = 0.0

        self.epsilon = epsilon
        self.epsilon_decay_rate = 0.999
        self.min_epsilon = 0.01
        self.n = n
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma

        self.summary_writer = summary_writer
        self.env_params = env_params

    def learn(self):
        """Perform DynaQ learning, return cumulative return"""
        s, _ = self.env.reset()  # initialize first state
        s = np.rint(s * self.grid_size)
        s = [int(i) for i in s]
        cum_reward = 0  # cumulative reward
        step = 1
        done = False

        while not done and step != self.max_steps:
            # Epsilon greedy action
            if np.random.random() < self.epsilon:
                a = self.env.action_space.sample()
            else:
                # combine action scores and q values
                q_values = self.q[tuple(s)]
                action_scores = self.model.get_action_scores(s)
                p_scores = q_values + action_scores
                a = np.argmax(p_scores)

            # Take action, observe outcome
            s_prime, r, done, _, _ = self.env.step(a)
            # self.env.render()
            s_prime = np.rint(s_prime * self.grid_size)
            s_prime = [int(i) for i in s_prime]
            unbounded, _ = self.model.get_action(s_prime)
            if unbounded:
                # revert to boundary
                if not (0 <= s_prime[0] <= self.grid_size - 1):
                    s_prime[0] = self.grid_size - 1
                if not (0 <= s_prime[1] <= self.grid_size - 1):
                    s_prime[1] = self.grid_size - 1

            # Q-Learning
            idx = tuple(np.concatenate([s, [int(a)]]))
            update = r + self.gamma * np.max(self.q[tuple(s_prime)]) - self.q[idx]
            if step == 1:
                self.avg_update = update
            else:
                self.avg_update = 0.05 * update + 0.95 * self.avg_update

            self.q[idx] += self.alpha * (update)

            # Check if update > theta, if yes: push to PQueue
            if self.p_thresh_upper >= abs(update) >= self.p_thresh_lower:
                self.p_queue.add((abs(update), (tuple(s), a)))

            # Learn model
            self.model.add(s, a, s_prime, r)

            # Planning for n steps
            self.planning()

            # Set state for next loop
            s = s_prime

            # Reset game if at the end
            if done:
                s, _ = self.env.reset()

            # Add reward to count
            cum_reward += r
            step += 1

            # Update PQueue thresholds
            self.p_thresh_lower = sorted(
                [
                    1.0,
                    self.p_thresh_lower,
                    (1 - self.update_threshold) * abs(self.avg_update),
                ]
            )[1]

        # update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.min_epsilon)
        return (
            cum_reward,
            step,
            (self.p_thresh_lower, abs(self.avg_update), self.p_thresh_upper),
        )

    def planning(self):

        for _ in range(self.n):
            if not len(self.p_queue):
                break
            # get random state and action in that state
            s, a = self.p_queue.pop()
            s_prime, r = self.model.step(s, a)
            update = (
                r + self.gamma * np.max(self.q[tuple(s_prime)]) - self.q[tuple(s + [a])]
            )
            self.q[
                tuple(
                    s
                    + [
                        a,
                    ]
                )
            ] += self.alpha * (update)

            # pick all neighbors of s, check their TD error and add to PQueue if > threshold
            _, actions = self.model.get_action(s, return_all=True)
            # cannot loop over all adjacent states as it can lead to an infinite back and forth between two states.
            actions = random.choice(actions)
            # backtrack s_bar from state with actions
            states = []
            idx = tuple(np.concatenate([s, [actions]]))
            idx = hash(idx)
            if idx in self.model.transitions:
                states.append(self.model.sample_s_prime(idx))

            if not states:
                break

            for state in states:
                _, actions = self.model.get_action(state, return_all=True)
                for a_i in actions:
                    idx = tuple(np.concatenate([state, [a_i]]))
                    if hash(idx) in self.model.transitions:
                        r_bar = self.model.rewards[hash(idx)]
                        update = abs(
                            r_bar + self.gamma * np.max(self.q[tuple(s)]) - self.q[idx]
                        )
                        if self.p_thresh_upper > abs(update) > self.p_thresh_lower:
                            self.p_queue.add((abs(update), (tuple(state), a_i)))


def objective(trial: optuna.Trial):

    alpha = trial.suggest_float("alpha", 0.0001, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.5, 0.99)
    epsilon = trial.suggest_float("epsilon", 0.1, 1.0, step=0.1)
    # planning_steps = trial.suggest_int("planning_steps", 10, 500, step=10)

    planning_steps = 1000  # planning loop terminates if priority queue is empty
    h_weight = trial.suggest_float("h_weight", 0.1, 1.0, step=0.1)

    env, env_params = make_env()

    max_steps = 2000
    max_episodes = 2000

    log_folder = "samples"
    summary_writer = tf.summary.create_file_writer(
        os.path.join(
            log_folder,
            f"alpha-{alpha}."
            + f"gamma-{gamma}."
            + f"epsilon-{epsilon}."
            + f"planning_steps-{planning_steps}."
            + f"max_episodes-{max_episodes}"
            + datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
    )
    dyna = DynaQ(
        env,
        planning_steps,
        alpha,
        gamma,
        epsilon,
        max_steps,
        env_params,
        summary_writer,
        h_weight,
    )
    cumulative_reward = 0.0
    for i in range(1, max_episodes):

        total_reward, num_steps, update_thresholds = dyna.learn()
        q_table = dyna.q
        q_table = tf.reduce_mean(q_table, 2)
        q_table = tf.divide(
            tf.subtract(q_table, tf.reduce_min(q_table)),
            tf.subtract(tf.reduce_max(q_table), tf.reduce_min(q_table)),
        )
        q_table = tf.reshape(q_table, (1,) + q_table.shape + (1,))
        with summary_writer.as_default(step=i):
            tf.summary.scalar("Episodic Reward", total_reward)
            tf.summary.image("Q_table", q_table)
            tf.summary.scalar("Total Steps", num_steps)
            tf.summary.scalar("Average Q", np.average(q_table))
            tf.summary.scalar(
                "Update_threshold/lower", np.average(update_thresholds[0])
            )
            tf.summary.scalar(
                "Update_threshold/average_update", np.average(update_thresholds[1])
            )
            tf.summary.scalar(
                "Update_threshold/upper", np.average(update_thresholds[2])
            )

        print(
            f"Episode: {i:4d} | Reward: {total_reward:6.2f} | Epsilon: {dyna.epsilon:4.2f} | Num-Steps: {num_steps:4d}"
        )
        cumulative_reward += total_reward

    env.close()
    return cumulative_reward


if __name__ == "__main__":

    import joblib

    log_folder = "samples"
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )
    study.optimize(objective, n_trials=500, show_progress_bar=True, n_jobs=4)
    joblib.dump(study, "optuna_study.pkl")
