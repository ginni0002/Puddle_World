import heapq
from collections import deque
import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).

import numpy as np
import keras
import tensorflow as tf
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import tensor_spec


class PQueue:

    def __init__(self):
        self._queue = []

        # {content: priority}
        self._entries = {}

    def __len__(self):
        return len(self._queue)

    def add(self, item):
        """
        Inserts item into heap
        item should be a tuple: (priority, content)
        """
        assert (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], float | int)
        ), "Item should be a tuple of length 2, (priority: int, content: any)"

        # since heapq implements min-heap, invert priority value
        priority, content = item
        priority = -abs(priority)
        if content in self._entries:
            if priority < self._entries[content]:

                # TODO: implement faster search indexing of queue
                item = tuple([priority, content])
                arr = np.array(self._queue, dtype=object)[:, 1]
                arr = np.array([hash(i) for i in arr])
                idx = int(np.squeeze(np.where(arr == hash(content))))
                self._queue[idx] = item
                self._entries[content] = priority
        else:
            # insert item in heap and hashmap
            heapq.heappush(self._queue, tuple([priority, content]))
            self._entries[content] = priority

    def pop(self):
        """
        Pops item with highest priority
        """
        if len(self._queue):
            _, item = heapq.heappop(self._queue)
            self._entries.pop(item)
            item = list(item)
            for idx, i in enumerate(item):
                if isinstance(i, tuple):
                    item[idx] = list(i)
            return item


class QNetwork:

    def __init__(
        self,
        input_dims: tuple,
        output_dims: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        lambd: float = 0.5,
        dims: list = [32, 32],
        batch_size: int = 32,
        buffer_length: int = 10000,
    ):

        self.input_dims = input_dims
        self.n_output = output_dims
        self.lr = lr
        self.dims = dims
        self.batch_size = batch_size
        self.gamma = gamma
        self.lambd = lambd

        self.opt = tf.keras.optimizers.Adam(lr)
        self.loss = tf.keras.losses.Huber()
        self.model = self._build_model()

        self.data_spec = (
            tf.TensorSpec(
                [
                    None,
                ],
                tf.float32,
                "state",
            ),
            tf.TensorSpec([1], tf.float32, "action"),
            tf.TensorSpec([1], tf.float32, "reward"),
            tf.TensorSpec(
                [
                    None,
                ],
                tf.float32,
                "next_state",
            ),
        )
        self.replay_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
            tensor_spec.to_array_spec(self.data_spec),
            capacity=buffer_length * self.batch_size,
        )
        self.temp_buffer = deque()

    def _build_model(self):

        inp = tf.keras.layers.Input(
            shape=self.input_dims,
            batch_size=self.batch_size,
        )
        x = inp
        for dim in self.dims:
            x = tf.keras.layers.Dense(dim, activation=tf.nn.relu)(x)

        out = tf.keras.layers.Dense(self.n_output)(x)
        model = tf.keras.Model(inputs=[inp], outputs=[out])
        return model

    def collect_rollout(self, sars: tuple):

        tf.ensure_shape(sars, self.data_spec)
        self.temp_buffer.append(sars)
        if len(self.temp_buffer) == self.batch_size:
            # add batch dim to items and append to replay buffer
            item = tf.reshape(self.temp_buffer, (1, *tf.shape(self.temp_buffer)))
            self.replay_buffer.add_batch(item)

            # clear temp buffer
            self.temp_buffer.clear()

    def predict(self, state):

        tf.ensure_shape(state, self.data_spec[0])
        return self.model.predict(state)

    def learn(self):
        """
        Update QNetwork parameters given replay mini-batch
        """
        sample = self.replay_buffer.get_next(self.batch_size)

        # compute Q-targets and current Q values, get loss from TD error
        # take gradient over batch
        s = sample[:, 0]
        a = sample[:, 1]
        r = sample[:, 2]
        s_ = sample[:, 3]
        print(s)
        q_value = self.model.predict(s)
        q_target = r + self.gamma * np.max(self.model.predict(s_))

        loss = self.loss(q_target, q_value)
        grads = tf.gradients(loss, self.model.trainable_weights)
        grads = self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
        return grads
