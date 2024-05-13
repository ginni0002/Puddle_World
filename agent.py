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

from utils import PQueue


class DynaQ:
    def __init__(
        self, env, n, alpha, gamma, epsilon, max_steps, env_params, summary_writer
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
            self.grid_size, self.state_size, self.action_size, env_params
        )

        # Priority sampling
        self.p_queue = PQueue()
        # initial thresholds, will change as q values grow/shrink
        self.p_thresh_lower = 0.0
        self.p_thresh_upper = 0.0
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
                self.avg_update = 0.01 * update + 0.99 * self.avg_update

            self.q[idx] += self.alpha * (update)

            # Check if update > theta, if yes: push to PQueue
            if self.p_thresh_upper > abs(update) > self.p_thresh_lower:
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
            self.p_thresh_lower = (1 - self.update_threshold) * self.avg_update
            self.p_thresh_upper = (1 + self.update_threshold) * self.avg_update

        # update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.min_epsilon)
        return cum_reward, step

    def planning(self):

        while not len(self.p_queue) == 0:

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
            direction_vec, _ = self.model.get_direction_vec(s, ref="start")
            # cannot loop over all adjacent states as it can lead to an infinite back and forth between two states.
            actions = np.array(
                [
                    self.model.action_map["x"][direction_vec[0]],
                    self.model.action_map["y"][direction_vec[1]],
                ]
            )
            actions = random.choice(actions[actions > -1])
            # backtrack s_bar from state with actions
            states = []
            for a_i in [actions]:
                idx = tuple(np.concatenate([s, [a_i]]))
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


class Model:
    def __init__(self, grid_size, n_states, n_actions, env_params):

        # transitions[(s, a)] = {s_: n_occurences}
        self.transitions = {}
        self.rewards = {}
        self.n_states = n_states
        self.n_actions = n_actions
        self.state_dims = len(self.n_states)
        self.grid_size = grid_size
        self.hash_list = []
        self.start_pos = env_params["start"]
        self.goal_pos = env_params["goal"]
        self.noise = env_params["noise"]
        self.action_map = {"x": {-1: 0, 0: -1, 1: 1}, "y": {-1: 2, 0: -1, 1: 3}}
        self.prev_dist = np.sum(
            [
                np.abs(self.start_pos[i] - self.goal_pos[i])
                for i in range(len(self.start_pos))
            ]
        )

    def add(self, s, a, s_prime, r):
        """
        Insert/update transition and reward
        """
        idx = tuple(np.concatenate([s, [a]]))
        idx = hash(idx)
        if idx not in self.transitions:
            self.hash_list.append(idx)
            self.transitions[idx] = {tuple(s_prime): 1}
            self.rewards[idx] = 0.0
        else:
            s_prime = tuple(s_prime)
            if s_prime not in self.transitions[idx]:
                self.transitions[idx][s_prime] = 1
            else:
                self.transitions[idx][s_prime] += 1

            self.rewards[idx] = self.rewards[idx] * 0.9 + r * 0.1

    def step(self, s, a):
        """Return next_state and reward for state-action pair"""
        idx = tuple(np.concatenate([s, [a]]))
        idx = hash(idx)

        if idx not in self.transitions:
            # random previously visited state and action
            idx = random.choice(self.hash_list)

        # pick next state based on sampling distribution
        s_prime = self.sample_s_prime(idx)
        r = self.rewards[idx]
        return list(s_prime), r

    def sample_s_prime(self, idx):
        """
        Return next state sampled from categorical transition distribution
        """
        s_ = list(self.transitions[idx].keys())
        indices = list(range(len(s_)))
        counts = list(self.transitions[idx].values())
        probs = counts / np.sum(counts)

        return s_[np.random.choice(indices, 1, replace=False, p=probs)[0]]

    def get_action(self, s, return_all=False):
        """
        Return bounded action(s) for state s
        """
        x, y = s
        unbounded = False
        if x == 0 or x >= self.grid_size - 1:
            a = [2, 3, 1 - x // self.grid_size]
            unbounded = True
        elif y == 0 or y >= self.grid_size - 1:
            a = [0, 1, 3 - y // self.grid_size]
            unbounded = True
        else:
            a = list(range(self.n_actions))

        return unbounded, a if return_all else random.choice(a)

    def get_action_scores(self, s):
        """
        Return array of action scores based on heuristic
        Heuristic: distance between current state and goal state
        """
        # convert s to  [0., 1.]
        direction_vec, dist = self.get_direction_vec(s)
        a_towards_goal = [
            self.action_map["x"][direction_vec[0]],
            self.action_map["y"][direction_vec[1]],
        ]

        action_scores = [
            0.7 / (1e-3 + dist) if i in a_towards_goal else 0
            for i in range(self.n_actions)
        ]

        return action_scores

    def get_direction_vec(self, s, ref="goal"):
        """
        Return tuple(direction, distance) from start/goal
        """

        s = [i / 100 for i in s]
        if ref == "goal":
            pos = self.goal_pos
        else:
            pos = self.start_pos

        delta_x = pos[0] - s[0]
        delta_y = pos[1] - s[1]
        # delta-x or delta-y will never be 0 as goal has a 0.01 radius
        direction = [
            int(np.round(num / (np.abs(num) + 1e-5))) for num in [delta_x, delta_y]
        ]
        dist = np.sum([np.abs(i) for i in [delta_x, delta_y]])

        return direction, dist


def make_env(prev_env=None):

    x_goal, y_goal = [random.uniform(0.05, 0.95) for _ in range(2)]

    x_start, y_start = (1 - x_goal, 1 - y_goal)
    env_params = {
        "start": [x_start, y_start],
        "goal": [x_goal, y_goal],
        "noise": np.random.uniform(0.0001, 0.001),
    }
    env = None
    if prev_env:
        # only change positions
        env = gym.make(
            "PuddleWorld-v0",
            render_mode="human",
            start=env_params["start"],
            goal=env_params["goal"],
            noise=env_params["noise"],
            puddle_top_left=prev_env.puddle_top_left,
            puddle_width=prev_env.puddle_width,
        )

    else:
        num_puddles = int(random.uniform(3, 6))
        pos_range = list(np.arange(0.0, 1.0, 0.1))
        puddle_pos = []
        puddle_sizes = []
        for _ in range(num_puddles):
            puddle_pos.append(
                [
                    random.choice(
                        [
                            i
                            for i in pos_range
                            if i not in env_params["goal"] + env_params["start"]
                        ]
                    )
                    for _ in range(2)
                ]
            )

            puddle_sizes.append([random.uniform(0.05, 0.4) for _ in range(2)])
        env = gym.make(
            "PuddleWorld-v0",
            render_mode="human",
            start=env_params["start"],
            goal=env_params["goal"],
            noise=env_params["noise"],
            puddle_top_left=puddle_pos,
            puddle_width=puddle_sizes,
        )
    return (
        env,
        env_params,
    )


if __name__ == "__main__":

    alpha = 0.05
    gamma = 0.9
    epsilon = 1.0
    max_steps = 2000
    planning_steps = 10
    max_episodes = 1000

    json_file = "content/pw2.json"
    with open(json_file) as f:
        env_params = json.load(f)

    log_folder = "samples"
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    # env, env_params = make_env()
    env = gym.make(
        "PuddleWorld-v0",
        render_mode="human",
        start=env_params["start"],
        goal=env_params["goal"],
        noise=env_params["noise"],
        puddle_top_left=env_params["puddle_top_left"],
        puddle_width=env_params["puddle_width"],
    )
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
    )
    # env.reset()
    # env.render()
    for j in range(100):
        alpha = random.choice([0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0001])
        gamma = random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
        epsilon = random.choice(list(np.arange(0.0, 1.0, 0.1)))
        planning_steps = random.choice(list(range(1, 200, 50)))
        for i in range(1, max_episodes):
            total_reward, num_steps = dyna.learn()
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

            print(
                f"Episode: {i:4d} | Reward: {total_reward:6.2f} | Epsilon: {dyna.epsilon:4.2f} | Num-Steps: {num_steps:4d}"
            )
            # randomize start_pos and goal_pos
            env, env_params = make_env(env)
            dyna.env = env
            dyna.env_params = env_params
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

    env.close()
