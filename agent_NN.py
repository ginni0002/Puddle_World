from datetime import datetime
import os
import joblib
import time

import tensorflow as tf
import numpy as np
import optuna

from utils import PQueue, get_heatmap
from model import TransitionModelNN, make_env
from dqn import QNetwork


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
        self.state_size = [
            [i, j]
            for i, j in zip(env.observation_space.low, env.observation_space.high)
        ]

        self.action_size = self.env.action_space.n
        self.n_states = self.env.observation_space.shape[0]
        self.input_spec = (
            tf.TensorSpec(
                [self.n_states, 1],
                tf.float32,
                "state",
            ),
            tf.TensorSpec([], tf.int64, "action"),
            tf.TensorSpec([], tf.float32, "reward"),
            tf.TensorSpec(
                [self.n_states, 1],
                tf.float32,
                "next_state",
            ),
        )
        self.input_spec_shapes = [i.shape.as_list() for i in self.input_spec]

        self.q = QNetwork(self.n_states, self.action_size, self.input_spec)
        self.model = TransitionModelNN(
            self.n_states,
            self.state_size,
            self.action_size,
            env_params,
            h_weight,
            self.input_spec,
        )

        self.lambd = lambd

        # Priority sampling
        self.p_queue = PQueue()
        # initial thresholds, will change as q values grow/shrink
        # upper threshold is fixed to prevent agent from just updating puddle interactions
        self.p_thresh_lower = 0.01
        self.p_thresh_upper = 20
        self.update_threshold = 0.25
        self.avg_update = tf.Variable(0.0)

        self.epsilon = epsilon
        self.epsilon_decay_rate = 0.99
        self.min_epsilon = 0.01
        self.n = n
        self.max_steps = max_steps
        self.alpha = tf.Variable(alpha)
        self.gamma = tf.Variable(gamma)

        self.summary_writer = summary_writer
        self.env_params = env_params
        self.render = False

    def reset_env_params(self, env, env_params):
        self.env = env
        self.env_params = env_params
        self.model.set_env_params(env_params)

    @tf.function(
        input_signature=(
            tf.TensorSpec(
                [None, 1],
            ),
            tf.TensorSpec([], tf.int64),
            tf.TensorSpec([], tf.float32),
            tf.TensorSpec(
                [None, 1],
            ),
        )
    )
    def update_q(self, s, a, r, s_prime):

        s_prime = tf.cast(s_prime, tf.float32)
        s = tf.cast(s, tf.float32)

        s = tf.reshape(s, [self.n_states, 1])
        s_prime = tf.reshape(s_prime, [self.n_states, 1])

        q_max = tf.reduce_max(self.q.predict(s_prime))
        q_vals = self.q.predict(s)
        q = tf.gather(q_vals, a)
        update = r + self.gamma * q_max - q
        self.avg_update.assign(0.05 * update + 0.95 * self.avg_update)
        r = tf.reshape(r, [1, 1])
        a = tf.reshape(a, [1, 1])
        a = tf.cast(a, tf.float32)

        a = tf.broadcast_to(a, self.input_spec_shapes[0])
        r = tf.broadcast_to(r, self.input_spec_shapes[0])

        sample = tf.stack([s, a, r, s_prime])
        # add transition to buffers
        self.q.replay_buffer.collect_rollout(sample)
        self.model.replay_buffer.collect_rollout(sample)

        return update

    @tf.function(
        input_signature=(
            tf.TensorSpec([None, 1], tf.float32),
            tf.TensorSpec([], tf.int64),
            tf.TensorSpec([None, 1], tf.float32),
            tf.TensorSpec([], tf.int64),
        )
    )
    def _get_update_static(self, s, a, s_bar, a_bar):
        _, r_bar = self.model.step(s_bar, a_bar)
        update = self.update_q(s_bar, a, r_bar, s)
        is_update = self.p_thresh_upper > abs(update) > self.p_thresh_lower
        return update, is_update

    def choose_action(self, state, return_all=False):

        state = tf.cast(state, tf.float32)
        state = tf.reshape(state, [self.n_states, 1])
        if tf.random.uniform([], 0.0, 1.0) < self.epsilon:
            a = self.model.get_action(state, return_all)
        else:
            # combine action scores and q values
            state = tf.reshape(state, [self.n_states, 1])
            state = tf.cast(state, tf.float32)
            q_values = self.q.predict(state)
            action_scores = self.model.get_action_scores(state)
            p_scores = q_values + action_scores
            a = tf.argmax(p_scores)
        return a

    @tf.function(input_signature=(tf.TensorSpec([None, 1], tf.float32),))
    def propogate_state(self, s):
        # train step
        self.q.learn()

        # pick all neighbors of s, check their update value and add to PQueue if > threshold

        # get random action for state s, use action on s to get to a neighbouring state s_bar
        # get all available actions of s_bar except previously taken action and update
        action = self.model.get_action(s)
        s_bar, _ = self.model.step(s, action)
        actions = self.model.get_action(s_bar, return_all=True)
        # remove action taken in state s to get to s_bar to prevent infinite recursion
        inv_action = tf.where(tf.math.equal(self.model.inv_action_keys, action))
        inv_action = tf.squeeze(tf.gather(self.model.inv_action_values, inv_action))

        ne_actions = tf.math.not_equal(actions, inv_action)
        ne_actions.set_shape(
            [
                None,
            ]
        )
        actions = tf.boolean_mask(actions, ne_actions)

        return actions, s_bar

    def learn(self):
        """Perform DynaQ learning, return cumulative return"""
        s, _ = self.env.reset()  # initialize first state
        s = self.model.featurize(s)
        cum_reward = 0  # cumulative reward
        step = 1
        done = False

        while not done and step != self.max_steps:
            # t1 = time.time()
            # Epsilon greedy action
            a = self.choose_action(s)
            # Take action
            s_prime, r, done, _, _ = self.env.step(a.numpy())
            if self.render:
                self.env.render()
            s_prime = self.model.featurize(s_prime)

            # Q-value update
            s = tf.reshape(s, (self.n_states, 1))
            s = tf.cast(s, tf.float32)

            s_prime = tf.reshape(s_prime, (self.n_states, 1))
            s_prime = tf.cast(s_prime, tf.float32)
            update = self.update_q(s, a, r, s_prime).numpy()
            loss_q = self.q.learn()
            loss_model = self.model.learn()

            # Check if update > theta, if yes: push to PQueue
            if self.p_thresh_upper > abs(update) > self.p_thresh_lower:
                self.p_queue.add([abs(update), tf.squeeze(s), a])

            # Planning for n steps
            tf.map_fn(
                self.planning,
                tf.range(0, self.n, dtype=tf.int32),
                tf.TensorSpec([], tf.int32),
            )
            s = s_prime
            if done:
                s, _ = self.env.reset()

            # Add reward to count
            cum_reward += r
            step += 1

        # update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.min_epsilon)
        return (
            cum_reward,
            step,
            (self.p_thresh_lower, abs(self.avg_update), self.p_thresh_upper),
            loss_q,
            loss_model,
        )

    @tf.function
    def _update_pqueue(self, s, a, s_bar, a_bar):
        update, is_update = self._get_update_static(s, a, s_bar, a_bar)
        if is_update:
            # only tuple is hashable
            s_ = tf.squeeze(s_bar)
            a_ = tf.cast(a_bar, tf.int64)
            tf.py_function(
                func=self.p_queue.add,
                inp=[tf.math.abs(update), s_, a_],
                Tout=tf.float32,
            )
        return is_update

    @tf.function
    def planning(self, i):

        if len(self.p_queue):
            # get random state and action in that state
            s, a = tf.py_function(self.p_queue.pop())
            a = tf.cast(a, tf.int64)
            s = tf.reshape(s, (self.n_states, 1))
            s = tf.cast(s, tf.float32)
            s_prime, r = self.model.step(s, a)
            self.update_q(s, a, r, s_prime)

            actions, s_bar = self.propogate_state(s)

            tf.map_fn(
                lambda x: self._update_pqueue(s, a, s_bar, x),
                actions,
                fn_output_signature=tf.TensorSpec([], tf.bool),
            )
        return i


def objective(trial: optuna.Trial):
    try:
        alpha = 0.09  # trial.suggest_float("alpha", 0.01, 0.1, step=0.01)
        gamma = 0.7  # trial.suggest_float("gamma", 0.5, 0.99)
        # epsilon = trial.suggest_float("epsilon", 0.5, 1.0, step=0.1)
        lambd = trial.suggest_float("lambd", 0.1, 0.9, step=0.1)
        epsilon = 1.0
        planning_steps = 10  # planning loop terminates if priority queue is empty
        h_weight = trial.suggest_float("h_weight", 0.1, 1.0, step=0.1)
        env, env_params = make_env()

        max_steps = 1000
        max_episodes = 500

        log_folder = "samples"
        summary_writer = tf.summary.create_file_writer(
            os.path.join(
                log_folder,
                f"alpha-{alpha}."
                + f"gamma-{gamma}."
                + f"epsilon-{epsilon}."
                + f"lambda-{lambd: 4.2f}."
                + f"h_weight-{h_weight:4.2f}."
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
            lambd,
        )
        cumulative_reward = 0.0

        with tf.device("/GPU:0"):
            for i in range(1, max_episodes):
                t1 = time.time()
                total_reward, num_steps, update_thresholds, loss_q, loss_model = (
                    dyna.learn()
                )
                # if i % 10 == 0:
                #     dyna.render = True
                # else:
                #     dyna.render = False

                with summary_writer.as_default(step=i):
                    tf.summary.scalar("Episodic Reward", total_reward)
                    tf.summary.scalar("Total Steps", num_steps)
                    tf.summary.scalar(
                        "Update_threshold/lower", np.average(update_thresholds[0])
                    )
                    tf.summary.scalar(
                        "Update_threshold/average_update",
                        np.average(update_thresholds[1]),
                    )
                    tf.summary.scalar(
                        "Update_threshold/upper", np.average(update_thresholds[2])
                    )

                    tf.summary.scalar("Losses/QNet", loss_q)

                    tf.summary.scalar("Losses/Model", loss_model)

                print(
                    f"Episode: {i:4d} |",
                    f" Reward: {total_reward:6.2f} |",
                    f"Epsilon: {dyna.epsilon:4.2f} |",
                    f"Num-Steps: {num_steps:4d} |",
                    f"Time/Episode: {time.time() - t1:4.2f}",
                )
                cumulative_reward += total_reward

        # env reset
        env, env_params = make_env()
        dyna.reset_env_params(env, env_params)

        env.close()
        return cumulative_reward

    except KeyboardInterrupt as e:
        trial.study.stop()


if __name__ == "__main__":

    log_folder = "samples"
    model_dir = "agents"
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    study = optuna.create_study(
        study_name="optuna_studies",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
    )

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
        )

    try:
        study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=1)

    finally:
        joblib.dump(study, "optuna_study.pkl")
    # objective()
