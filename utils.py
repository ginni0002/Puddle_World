import heapq
import numpy as np
import tensorflow as tf
import keras
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.agents import DqnAgent


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


# class QNetwork:

#     def __init__(
#         self,
#         input_dims: tuple,
#         output_dims: int,
#         lr: float = 0.001,
#         dims: list = [32, 32],
#         batch_size: int = 32,
#         buffer_length: int = 10000
#     ):

#         self.input_dims = input_dims
#         self.n_output = output_dims
#         self.lr = lr
#         self.dims = dims
#         self.batch_size = batch_size
        
#         self.opt = keras.optimizers.Adam(lr)
#         self.loss = keras.losses.Huber()
#         self.model = self._build_model()

#         self.data_spec = (
#             tf.TensorSpec([None,], tf.float32, 'state'),
#             tf.TensorSpec([1], tf.float32, 'action'),
#             tf.TensorSpec([1], tf.float32, 'reward'),
#             tf.TensorSpec([None,], tf.float32, 'next_state')
#         )
#         self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#             self.data_spec,
#             batch_size=batch_size,
#             max_length=buffer_length
#         )

#         self.gamma = 0.99

#     def _build_model(self):

#         model = keras.Sequential(
#             [
#                 keras.layers.Input(
#                     shape=self.input_dims,
#                     batch_size=self.batch_size,
#                 ),
#                 *[keras.layers.Dense(dim, activation=tf.nn.relu) for dim in self.dims],
#                 keras.layers.Dense(self.n_output)

#             ]
#         )
#         model.compile(optimizer=self.opt, loss=self.loss)
#         return model
    
#     def collect_rollout(self, sars: tuple):

#         tf.ensure_shape(sars, self.data_spec)
#         self.replay_buffer.add_batch(sars)

#     def learn(self):
#         """
#         Update QNetwork parameters given replay mini-batch
#         """
#         sample = self.replay_buffer.get_next(self.batch_size)
        
#         # compute Q-targets and current Q values, get loss from TD error
#         # take gradient over batch
#         for (s, a, r, s_) in sample:
#             q_value = self.model.predict(s)
#             q_max = np.max(self.model.predict(s_))
#             td_error = np.square(r + self.gamma * q_max - q_value)

#             self.model.compute_loss()


class DQNAgent:
    def __init__(self, input_dims: tuple,
        output_dims: int,
        lr: float = 0.001,
        dims: list = [32, 32],
        batch_size: int = 32,
        buffer_length: int = 10000):

        self.input_dims = input_dims
        self.n_output = output_dims
        self.lr = lr
        self.dims = dims
        self.batch_size = batch_size
        
        self.opt = keras.optimizers.Adam(lr)
        self.loss = keras.losses.Huber()
        self.model = self._build_model()

        self.data_spec = (
            tf.TensorSpec([None,], tf.float32, 'state'),
            tf.TensorSpec([1], tf.float32, 'action'),
            tf.TensorSpec([1], tf.float32, 'reward'),
            tf.TensorSpec([None,], tf.float32, 'next_state')
        )
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.data_spec,
            batch_size=batch_size,
            max_length=buffer_length
        )

        self.gamma = 0.99

    def _build_model(self):

        model = keras.Sequential(
            [
                keras.layers.Input(
                    shape=self.input_dims,
                    batch_size=self.batch_size,
                ),
                *[keras.layers.Dense(dim, activation=tf.nn.relu) for dim in self.dims],
                keras.layers.Dense(self.n_output)

            ]
        )
        model.compile(optimizer=self.opt, loss=self.loss)
        return model