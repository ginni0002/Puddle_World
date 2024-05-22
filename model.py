import random
import time
from utils import ReplayBuffer, random_choice

import gym_puddle
import gymnasium as gym


import tensorflow as tf
import numpy as np


class BaseModel:

    def __init__(self, n_states, state_size, n_actions, env_params, h_weight):
        self.n_states = n_states
        self.n_actions = n_actions
        self.state_size = state_size
        self.state_dims = len(self.state_size)
        self.set_env_params(env_params)

        self.action_map = {"x": {-1: 0, 0: -1, 1: 1}, "y": {-1: 2, 0: -1, 1: 3}}
        self.inv_action_keys = tf.range(0, self.n_actions, dtype=tf.int64)
        self.inv_action_values = tf.constant([1, 0, 3, 2], tf.int64)
        self.h_weight = h_weight

    def set_env_params(self, env_params):
        self.start_pos = env_params["start"]
        self.goal_pos = env_params["goal"]
        self.noise = env_params["noise"]

    def get_direction_vec(self, s, ref="goal"):
        """
        Return tuple(direction, distance) from start/goal
        """

        # TODO: Use state abstraction to get scores
        # normalize s to range [0., 1.]

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

    def get_action_scores(self, s):
        """
        Return array of action scores based on heuristic
        Heuristic: distance between current state and goal state
        """
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

    def step(self, s, a):
        raise NotImplementedError


class ModelTabular(BaseModel):
    def __init__(
        self, grid_size, n_states, state_size, n_actions, env_params, h_weight
    ):

        super().__init__(n_states, state_size, n_actions, env_params, h_weight)
        # transitions[(s, a)] = {s_prime: n_occurences}
        self.transitions = {}
        self.rewards = {}
        self.grid_size = grid_size
        self.hash_list = []

    def set_env_params(self, env_params):
        self.start_pos = env_params["start"]
        self.goal_pos = env_params["goal"]
        self.noise = env_params["noise"]

    def add(self, s, a, r, s_prime):
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
        s_prime = list(self.transitions[idx].keys())
        indices = list(range(len(s_prime)))
        counts = np.log(list(self.transitions[idx].values())) + 1e-3
        probs = counts / np.sum(counts)
        return s_prime[np.random.choice(indices, len(probs), replace=False, p=probs)[0]]

    def get_action(self, s, return_all=False):
        """
        Return bounded action(s) for state s
        """

        x, y = s
        unbounded = False
        if not (0.0 <= x <= self.grid_size - 1):
            # if at the vertical edges, movement restricted in x direction
            # 1-x/grid_size -> returns 0 if x at 1 and 1 if x at 0
            a = [2, 3, 1 - x // self.grid_size]
            unbounded = True

        elif 0.0 <= y <= self.grid_size - 1:
            a = [0, 1, 3 - y // self.grid_size]
            unbounded = True
        else:
            a = list(range(self.n_actions))

        return unbounded, a if return_all else random.choice(a)


class TransitionModelNN(BaseModel):

    def __init__(
        self, n_states, state_size, n_actions, env_params, h_weight, input_spec
    ):
        super().__init__(n_states, state_size, n_actions, env_params, h_weight)

        self.batch_size = 32
        self.dims = [128, 128]
        self.lr = 1e-3
        self.model = self._build_model()
        self.opt = tf.keras.optimizers.Adam(self.lr)
        self.loss = tf.keras.losses.Huber()
        self.model.compile(self.opt, self.loss, metrics=["accuracy"])

        self.spec_shapes = input_spec
        self.replay_buffer = ReplayBuffer(input_spec)
        self.a = tf.range(0, self.n_actions, dtype=tf.int32)
        self.action_mask = tf.ones([self.a.shape[0], 1], tf.int32)

    def _build_model(self):

        state = tf.keras.layers.Input(shape=[self.n_states, 1], name="state_inp")
        action = tf.keras.layers.Input(shape=[self.n_states, 1], name="action_inp")

        state_ = tf.keras.layers.Dense(
            self.dims[0], activation=tf.nn.relu, name="state"
        )(state)
        action_ = tf.keras.layers.Dense(
            self.dims[0], activation=tf.nn.relu, name="action"
        )(action)
        x = tf.keras.layers.Concatenate(name="concat")([state_, action_])

        for i, dim in enumerate(self.dims):
            x = tf.keras.layers.Dense(dim, activation=tf.nn.relu, name=str(i))(x)

        # next_state and reward output layers share the hidden layers
        x = tf.keras.layers.Flatten(name="flatten")(x)
        next_state = tf.keras.layers.Dense(self.n_states, name="next_state")(x)
        reward = tf.keras.layers.Dense(1, name="reward")(x)
        return tf.keras.Model(
            inputs={"state": state, "action": action},
            outputs={"next_state": next_state, "reward": reward},
        )

    def featurize(self, x):
        return x

    @tf.function(
        input_signature=(
            tf.TensorSpec(
                [
                    None,
                ]
            ),
            tf.TensorSpec(
                [
                    None,
                ]
            ),
        )
    )
    def _get_bounded_state(self, dim, s_dim):
        if not tf.searchsorted(dim, s_dim)[0] == 1:
            # revert s_prime to nearest boundary
            s_dim = tf.clip_by_value(s_dim, *self.state_size[0])
        return s_dim

    @tf.function(
        input_signature=(
            tf.TensorSpec([None, 1], dtype=tf.float32),
            tf.TensorSpec([], dtype=tf.int64),
        ),
    )
    def step(self, s, a):
        """
        Predict next state and reward
        """

        a = tf.cast(a, tf.float32)
        a = tf.broadcast_to(a, tf.shape(s))

        res = self.model({"state": tf.transpose(s), "action": tf.transpose(a)})
        s_prime = tf.transpose(res["next_state"])
        r = tf.squeeze(res["reward"])
        state_dims = tf.cast(self.state_size, tf.float32)
        s_prime = tf.vectorized_map(
            lambda x: self._get_bounded_state(x[0], x[1]), (state_dims, s_prime)
        )
        s_prime = tf.reshape(s_prime, tf.shape(s))
        return s_prime, r

    @tf.function(
        input_signature=(
            tf.TensorSpec([None, 1], dtype=tf.float32),
            tf.TensorSpec([], dtype=tf.bool),
        )
    )
    def get_action(self, s, return_all=tf.constant(False, tf.bool)):
        """
        Return action list or singular action given state s
        """
        a = self._get_action_static(s)
        if tf.math.equal(len(a), 0):
            a = tf.cast(0, tf.int64)
        else:

            if return_all:
                a = tf.reshape(
                    a,
                    [self.n_actions, ],
                )
                a = tf.cast(a, tf.int64)
            else:
                a, _ = random_choice(a, 1)
                a = tf.squeeze(a)
                a = tf.cast(a, tf.int64)

        return a

    @tf.function
    def _apply_mask(self, mask, indices, updates):
        return tf.tensor_scatter_nd_update(mask, tf.cast(indices, tf.int32), updates)

    @tf.function(input_signature=(tf.TensorSpec([None, 1], tf.float32),))
    def _get_action_static(self, s):
        """
        Return bounded action(s) for state s
        """

        x, y = tf.unstack(s, num=self.n_states, axis=0)

        # clip x and y and create mask with all values 1 except where clipped
        x_clip = tf.clip_by_value(x, *self.state_size[0])
        y_clip = tf.clip_by_value(y, *self.state_size[1])

        mask = tf.identity(self.action_mask)
        for idx, (pos, bound) in enumerate(zip([x_clip, y_clip], self.state_size)):
            low = tf.math.not_equal(pos, bound[0])
            high = tf.math.equal(pos, bound[1])

            indices = tf.constant([[idx * 2], [idx * 2 + 1]], tf.int32)
            updates = [low, high]
            mask = self._apply_mask(mask, indices, updates)

        # apply mask to remove clipped values
        a = tf.boolean_mask(tf.reshape(tf.identity(self.a), [self.n_actions, 1]), mask)
        a = tf.cast(a, tf.double)
        return a

    @tf.function
    def learn(self):
        """
        Retrieve sars' batches from QNet replay buffer (reuse) and train model
        based on transition history
        """
        sample = self.replay_buffer.get_random_samples()
        # compute Q-targets and current Q values, get loss from TD error
        # take gradient over batch
        sample = tf.squeeze(tf.stop_gradient(sample))
        s, a, r, s_prime = tf.unstack(sample, axis=1)

        # actions are indices, cast float -> int
        a, _ = tf.unstack(tf.cast(a, tf.int32), axis=1)
        r, _ = tf.unstack(r, axis=1)
        a = tf.expand_dims(a, -1)
        a = tf.broadcast_to(a, tf.shape(s))

        with tf.GradientTape() as tape:
            res = self.model({"state": s, "action": a})
            next_state_predict = res["next_state"]
            reward_predict = res["reward"]
            loss = self.loss(s_prime, next_state_predict) + self.loss(r, reward_predict)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss


def make_env(prev_env=None):
    """
    Make puddle-world env with random initialization
    limited to randomizing position if previous env provided
    """
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
        puddle_top_left = prev_env["puddle_top_left"]
        puddle_width = prev_env["puddle_width"]

    else:
        num_puddles = int(random.uniform(3, 6))
        pos_range = list(np.arange(0.0, 1.0, 0.1))
        puddle_top_left = []
        puddle_width = []
        for _ in range(num_puddles):
            puddle_top_left.append(
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

            puddle_width.append([random.uniform(0.05, 0.4) for _ in range(2)])

    env = gym.make(
        "PuddleWorld-v0",
        start=env_params["start"],
        goal=env_params["goal"],
        noise=env_params["noise"],
        puddle_top_left=puddle_top_left,
        puddle_width=puddle_width,
    )
    return (
        env,
        env_params,
    )


# def make_env():

#     import json

#     with open("content/pw1.json") as f:
#         env_params = json.load(f)

#     env = gym.make("PuddleWorld-v0", render_mode="human", **env_params)

#     return env, env_params


if __name__ == "__main__":

    import timeit

    env, env_params = make_env()
    state_size = [
        [i, j] for i, j in zip(env.observation_space.low, env.observation_space.high)
    ]
    model = TransitionModelNN(2, state_size, 4, env_params, 0.1)
    s = tf.constant([0.1, 0.4], tf.float32, [2, 1])
