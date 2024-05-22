import heapq
import numpy as np
import io

import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from tf_agents.replay_buffers import tf_uniform_replay_buffer


class PQueue:

    def __init__(self, shapes, types, maxlen=10000):
        self._queue = []

        self.shapes = shapes
        self.types = types

        # Note that entries should be lists as the priority needs to be mutable
        # {content: priority}
        self.maxlen = tf.cast(maxlen, tf.int32)

        self._entries = tf.lookup.experimental.DenseHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.float32,
            default_value=tf.constant((0.), tf.float32),
            empty_key=0,
            deleted_key=-100,
        )

        self.default_value = []
        for shape, type in zip(shapes, types):
            self.default_value.append(tf.random.uniform(shape, maxval=1, dtype=type))

    @tf.function(
        input_signature=(
            [tf.TensorSpec([None, 1], tf.float32), tf.TensorSpec([], tf.int64)],
        )
    )
    def _get_hashed(self, content):

        c_new = list(
            map(
                lambda x: (
                    tf.expand_dims(tf.cast(x, tf.float32), -1)
                    if len(tf.shape(x)) == 0
                    else tf.squeeze(tf.cast(x, tf.float32))
                ),
                content,
            )
        )

        tensor_to_string = tf.strings.as_string(tf.concat(c_new, 0))
        hashed_values = tf.strings.to_hash_bucket_fast(tensor_to_string, 1 << 10)
        return tf.reduce_sum(hashed_values)

    @tf.function
    def size(self):
        return len(self._queue)

    def add(self, priority, content):
        """
        Inserts item into heap
        item should be a list: [priority, contents (spread out instead of a list)]
        """

        def update_item(c_ref, c):
            # if item in PQueue and higher priority, replace previous priority in heap and hashmap
            prev_priority = self._entries.lookup(c_ref)
            tf.cond(
                tf.cast(priority < prev_priority, tf.bool),
                lambda: true_func2(c_ref, c),
                lambda: False,
            )

        def true_func2(c_ref, c):
            heapq.heapify(self._queue)
            self._entries.insert_or_assign(c_ref, priority)
            return True

        def create_item(c_ref, c):
            # insert item in heap and hashmap

            # check if heap is filled
            # compare last element of heap if full and replace/keep last element.
            if tf.equal(self.size(), self.maxlen):
                p_last, *c_last = self._queue[-1]
                if priority < p_last:
                    self._queue[-1] = [priority, *c]
                    self._entries.insert_or_assign(c_ref, priority)

                    # remove previous entry
                    c_ref_last = self._get_hashed(c_last)
                    self._entries.erase(tf.expand_dims(c_ref_last, -1))

                # to maintain tf execution
                else:
                    self._entries.insert_or_assign(c_last, p_last)

            else:
                heapq.heappush(self._queue, [priority, *c])
                self._entries.insert_or_assign(c_ref, priority)

        # since heapq implements min-heap, invert priority value
        priority = -abs(priority)

        # tf priority queues take int priority only
        priority = tf.cast(priority, tf.float32).numpy()
        content_ref = self._get_hashed(content)

        # reshape content to match Priority Queue shapes
        content = list(map(lambda c: tf.squeeze(tf.cast(c, tf.float32)), content))

        # if lookup returns non-default value -> item exists, check priority and update
        tf.cond(
            tf.cast(
                tf.math.not_equal(
                    self._entries.lookup(content_ref), self._entries._default_value
                ),
                tf.bool,
            ),
            lambda: update_item(content_ref, content),
            lambda: create_item(content_ref, content),
        )

        return self.maxlen

    def pop(self):
        """
        Pops item with highest priority
        """

        if tf.math.not_equal(self.size(), tf.constant(0, tf.int32)):
            _, content = heapq.heappop(self._queue)
            content_ref = self._get_hashed(content)

            self._entries.erase(tf.expand_dims(content_ref, -1))
            item = content
            for idx, i in enumerate(content):
                if isinstance(i, tuple):
                    item[idx] = list(i)
            return item

        else:

            return self.default_value


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
