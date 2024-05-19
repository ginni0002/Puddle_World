import heapq
import time
import io
from collections import deque

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt


class PQueue:

    def __init__(self, maxlen=100):
        self._queue = []

        # {content: priority}
        self._entries = {}
        self.maxlen = maxlen

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
            and isinstance(item[0], float | np.float32 | int)
        ), "Item should be a tuple of length 2, (priority: float | int, content: any)"

        # since heapq implements min-heap, invert priority value
        priority, content = item
        priority = -abs(priority)
        if content in self._entries:
            if priority < self._entries[content]:
                item = tuple([priority, content])
                arr = np.array(self._queue, dtype=object)[:, 1]
                idx = np.array(list(map(lambda elem: elem == content, arr)))
                idx = np.squeeze(np.where(np.squeeze(idx)))[()]
                self._queue[idx] = item
                self._entries[content] = priority
        else:
            # insert item in heap and hashmap

            # check if heap is filled
            # compare last element of heap if full and replace/keep last element.
            if len(self._queue) == self.maxlen:
                p_last, c_last = self._queue[-1]
                if priority < p_last: 
                    self._queue[-1] = tuple([priority, content])
                    self._entries[content] = priority

                    # remove previous entry
                    self._entries.pop(c_last)

            else:
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


class ReplayBuffer:
    def __init__(self, maxlen=10000, sample_spec=None):

        self.maxlen = maxlen
        self.replay_buffer = deque([], maxlen)

        # both model and dqn use same input spec
        # TODO: replace hardcoded input spec
        input_dims = 2

        if not sample_spec:
            self.sample_spec = (
                tf.TensorSpec(
                    [input_dims, 1],
                    tf.float32,
                    "state",
                ),
                tf.TensorSpec([], tf.int64, "action"),
                tf.TensorSpec([], tf.float32, "reward"),
                tf.TensorSpec(
                    [input_dims, 1],
                    tf.float32,
                    "next_state",
                ),
            )
        else:
            self.sample_spec = sample_spec

        self.spec_shapes = [i.shape.as_list() for i in self.sample_spec]

    def __len__(self):
        return len(self.replay_buffer)

    def as_list(self):
        return list(self.replay_buffer)

    def collect_rollout(self, sample):
        """
        Collect samples for the replay buffer for batch updates
        """
        try:
            sample = [tf.broadcast_to(i, max(self.spec_shapes)) for i in sample]
            sample = tf.expand_dims(sample, -1)
        except ValueError:
            print(f"Invalid (s, a, r, s') tuple: {sample}")
            print(f"Valid shapes: {self.spec_shapes}")
            raise

        self.replay_buffer.append(sample)

    @tf.function
    def get_random_samples(self, batch_size=32):
        """
        Generates a batch of indices from a uniform distribution
        """

        indices = tf.random.uniform(
            (batch_size,), 0, len(self.replay_buffer), dtype=tf.int32
        )
        return tf.gather(tf.identity(list(self.replay_buffer)), indices)


def get_heatmap(matrix):
    """
    Plots 2-D matrix as a heatmap
    Returns: matplotlib.ax object
    """

    mat_dims = tf.shape(matrix)
    if len(mat_dims) > 2:
        # print("Warning: Matrix rank > 2 provided, last (n-2) dims reduced")
        # reduce last n-2 dims
        matrix = tf.reduce_mean(matrix, axis=[range(-len(mat_dims) + 2, 0)])

    vmin = tf.reduce_min(matrix)
    vmax = tf.reduce_max(matrix)
    hmap = sns.heatmap(matrix, vmin=vmin, vmax=vmax)

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(hmap.get_figure())
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


if __name__ == "__main__":

    num_ep = 1000
    s = (0.0, 1.0)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=512)]
        )

    from model import Model, make_env

    env, env_params = make_env()
    grid_size = 40
    state_size = [
        int(i)
        for i in (env.observation_space.high - env.observation_space.low) * grid_size
    ]
    q_net = QNetwork(2, 4)
    model = Model(grid_size, state_size, 4, env_params, 0.5)
    env.reset()
    try:
        for i in range(num_ep):

            a = q_net.predict(tf.reshape(s, q_net.spec_shapes[0]))
            a = np.argmax(a)
            s_, r, _, _, _ = env.step(a)
            a = tf.reshape(a, (1,))
            r = tf.reshape(r, (1,))
            q_net.collect_rollout((s, a, r, s_))
            if i == 0:
                t1 = time.time()
            if i == q_net.batch_size:
                q_net.learn()
        print(time.time() - t1)
    except Exception as e:
        raise e
