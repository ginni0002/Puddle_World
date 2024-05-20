import heapq
import time
import io
from collections import defaultdict

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from tf_agents.replay_buffers import tf_uniform_replay_buffer


class PQueue:

    def __init__(self, maxlen=100):
        self._queue = []

        # Note that entries should be lists as the priority needs to be mutable
        # {content: priority}
        self._entries = defaultdict(int)
        self.maxlen = maxlen
        self.sample_value = [
            tf.constant([0.1, 0.1]),
            1.0
        ]

    def __len__(self):
        return len(self._queue)

    def _get_hashed(self, content):
        content = list(
            map(
                lambda x: (
                    tf.cast(tf.expand_dims(x, -1), tf.float32)
                    if len(tf.shape(x)) == 0
                    else tf.cast(x, tf.float32)
                ),
                content,
            )
        )
        c_hashed = tf.concat(content, 0)
        c = hash(str(c_hashed).encode())
        return c, content

    def add(self, *item):
        """
        Inserts item into heap
        item should be a list: [priority, contents (spread out instead of a list)]
        """
        while len(item) == 1:
            item = item[0]
        assert (
            isinstance(item, list | tuple)
            # and isinstance(item[0], tf.Tensor)
        ), f"Item should be a tuple or list, (priority: float | int, content: any), item: {item}"

        # since heapq implements min-heap, invert priority value
        priority, *content = item
        priority = -abs(priority)
        content_ref, content = self._get_hashed(content)

        if content_ref in self._entries:
            prev_priority = self._entries[content_ref]
            if priority < prev_priority:
                # if item in PQueue and higher priority, replace previous priority in heap and hashmap
                self._entries[content_ref] = priority

                heapq.heapify(self._queue)

            # to maintain tf execution
            else:
                self._entries[content_ref] = prev_priority
        else:
            # insert item in heap and hashmap

            # check if heap is filled
            # compare last element of heap if full and replace/keep last element.
            if len(self._queue) == self.maxlen:
                p_last, c_last = self._queue[-1]
                if priority < p_last:
                    self._queue[-1] = list([priority, content])
                    self._entries[content_ref] = priority

                    heapq.heapify(self._queue)

                    # remove previous entry
                    c_last_ref, _ = self._get_hashed(c_last)
                    self._entries.pop(c_last_ref)

                # to maintain tf execution
                else:
                    self._entries[content_ref] = p_last

            else:
                heapq.heappush(self._queue, list([priority, content]))
                self._entries[content_ref] = priority

        return priority

    def pop(self):
        """
        Pops item with highest priority
        """
        if len(self._queue):
            _, content = heapq.heappop(self._queue)
            content_ref, _ = self._get_hashed(content)
            del self._entries[content_ref]
            item = list(
                map(
                    lambda x: (
                        tf.squeeze(x)
                        if len(tf.shape(x)) == 1
                        else tf.cast(x, tf.float32)
                    ),
                    content,
                )
            )
            for idx, i in enumerate(content):
                if isinstance(i, tuple):
                    item[idx] = list(i)
            return item

        else:
            return self.sample_value


class ReplayBuffer:
    """
    Tensorflow compatible custom replay buffer

    """

    def __init__(self, sample_spec, maxlen=10000, batch_size=32):

        self.maxlen = maxlen

        # both model and dqn use same input spec
        self.sample_spec = [
            tf.TensorSpec([len(sample_spec), *sample_spec[0].shape.as_list()])
        ]
        self.batch_size = batch_size

        self.spec_shapes = [i.shape.as_list() for i in self.sample_spec]
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.sample_spec[0], batch_size=1, max_length=maxlen
        )

    @tf.function
    def collect_rollout(self, sample):
        """
        Collect samples for the replay buffer for batch updates
        """
        try:
            # add batch dim to items and append to replay buffer
            self.replay_buffer.add_batch(tf.expand_dims(sample, 0))
        except ValueError:
            print(f"Invalid (s, a, r, s') tuple: {sample}")
            print(f"Valid shapes: {self.spec_shapes}")
            raise

    @tf.function
    def get_random_samples(self):
        """
        Generates a batch of indices from a uniform distribution
        """

        sample_batch, _ = self.replay_buffer.get_next(self.batch_size)
        return sample_batch


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

# https://github.com/tensorflow/tensorflow/issues/8496
@tf.function
def random_choice(x, size, axis=0, unique=True):
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    sample = tf.gather(x, sample_index, axis=axis)

    return sample, sample_index


if __name__ == "__main__":

    a = tf.constant([1, 2])
    for _ in range(10):
        print(random_choice(a, 1)[0])
