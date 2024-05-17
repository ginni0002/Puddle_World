from datetime import datetime
import os
import joblib

import tensorflow as tf
import numpy as np
import optuna
import jax
import jax.numpy as jnp
from jax import jit

from utils import PQueue, QNetwork, get_heatmap
from model import Model, make_env

"""
    Static Helper functions for jax-jit 
"""


@jit
def set_update_threshold_static(th_low, n, len_p_queue, avg_update, th_update):

    # if alpha near 0, n > len_p_queue and vice versa
    alpha = jax.nn.sigmoid(jnp.float32(len_p_queue - n))
    th_new = alpha * th_low * 1.05 + (1 - alpha) * jnp.max(
        jnp.array([0.95 * th_low, 0.01])
    )
    return jnp.sort(
        jnp.array(
            [
                th_new,
                th_low,
                (1 - th_update) * abs(avg_update),
            ]
        )
    )[1]


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
        self.grid_size = 50
        self.state_size = [
            int(i)
            for i in (self.env.observation_space.high - self.env.observation_space.low)
            * self.grid_size
        ]

        self.action_size = self.env.action_space.n
        self.q = QNetwork(self.env.observation_space.shape[0], self.action_size)
        self.model = Model(
            self.grid_size, self.state_size, self.action_size, env_params, h_weight
        )
        self.visits = np.zeros(tuple(self.state_size))

        self.trace_dict = {}
        self.lambd = lambd

        # Priority sampling
        self.p_queue = PQueue()
        # initial thresholds, will change as q values grow/shrink
        # upper threshold is fixed to prevent agent from just updating puddle interactions
        self.p_thresh_lower = 0.01
        self.p_thresh_upper = 20
        self.update_threshold = 0.25
        self.avg_update = 0.0

        self.epsilon = epsilon
        self.epsilon_decay_rate = 0.99
        self.min_epsilon = 0.01
        self.n = n
        self.max_steps = max_steps
        self.alpha = alpha
        self.gamma = gamma

        self.summary_writer = summary_writer
        self.env_params = env_params
        self.render = False

    def reset_env_params(self, env, env_params):
        self.env = env
        self.env_params = env_params
        self.model.set_env_params(env_params)

    def set_update_threshold(self):

        # Update PQueue thresholds
        self.p_thresh_lower = set_update_threshold_static(
            self.p_thresh_lower,
            self.n,
            len(self.p_queue),
            self.avg_update,
            self.update_threshold,
        )

    def update_q(self, s, a, r, s_prime):
        q_max = jnp.max(self.q.predict(tf.reshape(s_prime, self.q.spec_shapes[0])))
        q_vals = self.q.predict(s)
        q = tf.gather(q_vals, a)
        update = r + self.gamma * q_max - q
        self.avg_update = 0.05 * update + 0.95 * self.avg_update
        s, a, r, s_ = [
            tf.reshape(i, j) for i, j in zip((s, a, r, s_prime), self.q.spec_shapes)
        ]
        self.q.collect_rollout((s, a, r, s_))
        return np.asarray(update)

    @tf.function(input_signature=(tf.TensorSpec([None, 1], dtype=tf.float32),))
    def choose_action(self, state):

        if tf.random.uniform([]) < self.epsilon:
            a = self.model.get_action(state)
        else:
            # combine action scores and q values
            q_values = self.q.predict(tf.reshape(s, self.q.spec_shapes[0]))
            action_scores = self.model.get_action_scores(s)
            p_scores = q_values + action_scores
            a = tf.argmax(p_scores)
        return a

    def learn(self):
        """Perform DynaQ learning, return cumulative return"""
        s, _ = self.env.reset()  # initialize first state
        s = self.model.featurize(s)
        cum_reward = 0  # cumulative reward
        step = 1
        done = False
        self.trace_dict = {}

        while not done and step != self.max_steps:
            # Epsilon greedy action
            a = self.choose_action(s)
            # Take action
            s_prime, r, done, _, _ = self.env.step(a)
            # if self.render:
            #     self.env.render()
            s_prime = self.model.featurize(s_prime)

            # Update eligibility traces
            idx = self.model.get_idx(s, a)
            trace_key = idx
            if trace_key not in self.trace_dict:
                self.trace_dict[trace_key] = 1
            else:
                self.trace_dict[trace_key] += 1

            # Q-value update
            update = self.update_q(s, a, r, s_prime)

            # Check if update > theta, if yes: push to PQueue
            if self.p_thresh_upper > abs(update) > self.p_thresh_lower:
                self.p_queue.add((abs(update), (tuple(s), a)))

            # Learn model
            self.model.add(s, a, s_prime, r)

            # Planning for n steps
            self.planning()

            s = s_prime
            if done:
                s, _ = self.env.reset()

            # Add reward to count
            cum_reward += r
            step += 1

            # Update the trace
            self.trace_dict[trace_key] *= self.gamma * self.lambd
            # Update PQueue thresholds
            self.set_update_threshold()

        # update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.min_epsilon)
        return (
            cum_reward,
            step,
            (self.p_thresh_lower, abs(self.avg_update), self.p_thresh_upper),
            len(self.p_queue),
            self.visits,
        )

    @tf.function
    def planning(n, p_queue):
        # t1 = time.time()
        for _ in range(self.n):

            if not len(self.p_queue):
                break
            # get random state and action in that state
            s, a = self.p_queue.pop()
            s_prime, r = self.model.step(s, a)
            idx = self.model.get_idx(s, a)
            update = self.update_q(s, a, r, s_prime)

            # pick all neighbors of s, check their TD error and add to PQueue if > threshold
            action = self.model.get_action(s)
            # cannot loop over all adjacent states as it can lead to an infinite back and forth between two states.
            # backtrack s_bar from state with actions

            idx = self.model.get_idx(s, action)
            if idx in self.model.transitions:
                s_bar = self.model.sample_s_prime(idx)
            else:
                continue

            actions = self.model.get_action(s_bar, return_all=True)
            for a_bar in actions:
                idx = self.model.get_idx(s, a)
                if idx in self.model.transitions:
                    r_bar = self.model.rewards[idx]
                    update = self.update_q(s_bar, a, r_bar, s)
                    if self.p_thresh_upper > abs(update) > self.p_thresh_lower:
                        self.p_queue.add((abs(update), (tuple(s_bar), a_bar)))
                        # self.visits[tuple(state)] += 1

            self.set_update_threshold()
        # print(time.time() - t1, self.p_thresh_lower, len(self.p_queue))


def objective(trial: optuna.Trial):

    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=512)]
            )

        alpha = 0.09  # trial.suggest_float("alpha", 0.01, 0.1, step=0.01)
        gamma = 0.7  # trial.suggest_float("gamma", 0.5, 0.99)
        # epsilon = trial.suggest_float("epsilon", 0.5, 1.0, step=0.1)
        lambd = trial.suggest_float("lambd", 0.1, 0.9, step=0.1)
        epsilon = 1.0
        planning_steps = 100  # planning loop terminates if priority queue is empty
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
                + f"lambda-{lambd}."
                + f"h_weight-{h_weight}."
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
        for i in range(1, max_episodes):
            total_reward, num_steps, update_thresholds, queue_length, visits = (
                dyna.learn()
            )
            q_table = get_heatmap(dyna.q)
            visits = get_heatmap(visits)
            if i % 10 == 0:
                dyna.render = True
            else:
                dyna.render = False

            with summary_writer.as_default(step=i):
                tf.summary.scalar("Episodic Reward", total_reward)
                tf.summary.image("Q_table", q_table)
                tf.summary.image("Visits", visits)
                tf.summary.scalar("Total Steps", num_steps)
                tf.summary.scalar("Average Q", np.average(dyna.q))
                tf.summary.scalar(
                    "Update_threshold/lower", np.average(update_thresholds[0])
                )
                tf.summary.scalar(
                    "Update_threshold/average_update", np.average(update_thresholds[1])
                )
                tf.summary.scalar(
                    "Update_threshold/upper", np.average(update_thresholds[2])
                )
                tf.summary.scalar(
                    "PQueue_length",
                    queue_length,
                )

            print(
                f"Episode: {i:4d} | Reward: {total_reward:6.2f} | Epsilon: {dyna.epsilon:4.2f} | Num-Steps: {num_steps:4d}"
            )
            cumulative_reward += total_reward

        # env reset
        env, env_params = make_env()
        dyna.reset_env_params(env, env_params)

        env.close()
        return cumulative_reward

    except KeyboardInterrupt:
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
    try:
        study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=1)
    finally:
        joblib.dump(study, "optuna_study.pkl")
