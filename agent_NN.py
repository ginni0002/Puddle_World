import random
import optuna.terminator
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

from utils import PQueue, QNetwork
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
        lambd,
    ):
        self.env = env
        self.grid_size = 40
        self.state_size = [
            int(i)
            for i in (self.env.observation_space.high - self.env.observation_space.low)
            * self.grid_size
        ]

        self.action_size = self.env.action_space.n
        self.q_net = QNetwork(self.env.observation_space.shape, self.action_size)
        self.model = Model(
            self.grid_size, self.state_size, self.action_size, env_params, h_weight
        )

        self.trace_dict = {}
        self.lambd = lambd

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
                q_values = self.q_net.predict(s)
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

            # Update eligibility traces
            idx = tuple(np.concatenate([s, [int(a)]]))
            trace_key = idx
            if trace_key not in self.trace_dict:
                self.trace_dict[trace_key] = 1
            else:
                self.trace_dict[trace_key] += 1

            # Q-Learning
            update = (
                r
                + self.gamma * np.max(self.q_net.predict(s_prime))
                - self.q_net.predict(s)
            )
            if step == 1:
                self.avg_update = update
            else:
                self.avg_update = 0.05 * update + 0.95 * self.avg_update

            self.q[idx] += self.alpha * self.trace_dict[trace_key] * update

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
            # Update the trace
            self.trace_dict[trace_key] *= self.gamma * self.lambd
            # Update PQueue thresholds
            self.p_thresh_lower = sorted(
                [
                    (
                        1.0 + (self.n - len(self.p_queue))
                        if self.n < len(self.p_queue)
                        else 1.0
                    ),
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
