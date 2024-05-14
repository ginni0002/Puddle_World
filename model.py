import numpy as np
import random

import gym_puddle
import gymnasium as gym

class Model:
    def __init__(self, grid_size, n_states, n_actions, env_params, h_weight):

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
        self.h_weight = h_weight

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
        counts = np.array(list(self.transitions[idx].values()))
        probs = counts / np.sum(counts)

        return s_[np.random.choice(indices, len(probs), replace=False, p=probs)[0]]

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
            self.h_weight / (1e-3 + dist) if i in a_towards_goal else 0
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


# def make_env(prev_env=None):

#     x_goal, y_goal = [random.uniform(0.05, 0.95) for _ in range(2)]

#     x_start, y_start = (1 - x_goal, 1 - y_goal)
#     env_params = {
#         "start": [x_start, y_start],
#         "goal": [x_goal, y_goal],
#         "noise": np.random.uniform(0.0001, 0.001),
#     }
#     env = None
#     if prev_env:
#         # only change positions
#         env = gym.make(
#             "PuddleWorld-v0",
#             render_mode="human",
#             start=env_params["start"],
#             goal=env_params["goal"],
#             noise=env_params["noise"],
#             puddle_top_left=prev_env.puddle_top_left,
#             puddle_width=prev_env.puddle_width,
#         )

#     else:
#         num_puddles = int(random.uniform(3, 6))
#         pos_range = list(np.arange(0.0, 1.0, 0.1))
#         puddle_pos = []
#         puddle_sizes = []
#         for _ in range(num_puddles):
#             puddle_pos.append(
#                 [
#                     random.choice(
#                         [
#                             i
#                             for i in pos_range
#                             if i not in env_params["goal"] + env_params["start"]
#                         ]
#                     )
#                     for _ in range(2)
#                 ]
#             )

#             puddle_sizes.append([random.uniform(0.05, 0.4) for _ in range(2)])
#         env = gym.make(
#             "PuddleWorld-v0",
#             render_mode="human",
#             start=env_params["start"],
#             goal=env_params["goal"],
#             noise=env_params["noise"],
#             puddle_top_left=puddle_pos,
#             puddle_width=puddle_sizes,
#         )
#     return (
#         env,
#         env_params,
#     )

def make_env():

    import json
    with open('content/pw1.json') as f:
        env_params = json.load(f)

    env = gym.make(
            "PuddleWorld-v0",
            render_mode="human",
            **env_params
        )
    
    return env, env_params