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

        # store priority as key and (s, a) as tensor value concatenated
        self._entries_content = tf.lookup.experimental.DenseHashTable(
            key_dtype=tf.string,
            value_dtype=tf.float32,
            default_value=tf.constant((0.0, 0.0, 0.0), tf.float32),
            empty_key="-101",
            deleted_key="-100",
        )

        self._entries_priority = tf.lookup.experimental.DenseHashTable(
            key_dtype=tf.string,
            value_dtype=tf.float32,
            default_value=tf.constant(100.0, tf.float32),
            empty_key="",
            deleted_key="!",
        )

        self.default_value = []
        for shape, type in zip(shapes, types):
            self.default_value.append(tf.random.uniform(shape, maxval=1, dtype=type))

    @tf.function(
        input_signature=(
            [tf.TensorSpec([None, 1], tf.float32), tf.TensorSpec([], tf.int64)],
        )
    )
    def _concat_tensors(self, content):
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
        return tf.concat(c_new, 0)

    # @tf.function(input_signature=(tf.TensorSpec([None, 1], tf.float32),))
    def _get_unstacked_tensors(self, content_concat):
        s, a = tf.split(content_concat, [self.shapes[0][0], self.shapes[1][0]])
        return [s, a]

    # @tf.function(
    #     input_signature=(
    #         [tf.TensorSpec([None, 1], tf.float32), tf.TensorSpec([], tf.int64)],
    #     )
    # )
    # def _get_hashed(self, content):

    #     c_new = self._concat_tensors(content)

    #     tensor_to_string = tf.strings.as_string(c_new)
    #     hashed_values = tf.strings.to_hash_bucket_fast(tensor_to_string, 1 << 10)
    #     return tf.reduce_sum(hashed_values)

    @tf.function
    def _update_hashmap(self, priority, c_ref, prev_priority=None):
        c_ref_val = tf.identity(c_ref)
        priority_val = tf.identity(priority)

        c_ref = str(c_ref)
        priority = str(priority)
        self._entries_content.insert_or_assign(priority, c_ref_val)
        self._entries_priority.insert_or_assign(c_ref, priority_val)

        if prev_priority:
            prev_priority = str(prev_priority)
            prev_c_ref = self._entries_content.lookup(prev_priority)
            self._entries_priority.erase(tf.expand_dims(prev_c_ref, -1))
            self._entries_content.erase(tf.expand_dims(prev_priority, -1))

    @tf.function
    def size(self):
        return len(self._queue)

    def add(self, priority, content):
        """
        Inserts item into heap
        item should be a list: [priority, contents (spread out instead of a list)]
        """

        def update_item(c_ref, priority):
            # if item in PQueue and higher priority, replace previous priority in heap and hashmap
            prev_priority = self._entries_priority.lookup(str(c_ref))
            tf.cond(
                tf.cast(priority < prev_priority, tf.bool),
                lambda: true_func2(c_ref, priority, prev_priority),
                lambda: False,
            )

        def true_func2(c_ref, priority, prev_priority):
            # get index of priority for prev content from PQueue and update PQueue priority -> content_new
            idx = tf.squeeze(
                tf.where(self._queue == tf.constant(prev_priority)), axis=0
            )[0]
            self._queue[idx] = priority
            heapq.heapify(self._queue)

            # update references in hashmap
            self._update_hashmap(priority, c_ref, prev_priority)

        def create_item(c_ref, priority):
            # insert item in heap and hashmap

            # check if heap is filled
            # compare last element of heap if full and replace/keep last element.
            if tf.equal(self.size(), self.maxlen):
                p_last = self._queue[-1]
                if priority < p_last:
                    heapq.heapreplace(self._queue, priority)
                    self._update_hashmap(priority, c_ref, p_last)

            else:
                # not filled, just insert and update
                heapq.heappush(self._queue, priority)
                self._update_hashmap(priority, c_ref)

        # since heapq implements min-heap, negate priority value
        priority = -priority

        # tf priority queues take int priority only
        priority = tf.cast(priority, tf.float32).numpy()
        content_ref = self._concat_tensors(content)
        # reshape content to match Priority Queue shapes
        # content = list(map(lambda c: tf.squeeze(tf.cast(c, tf.float32)), content))

        # if lookup returns non-default value -> item exists, check priority and update

        condt = tf.cast(
            tf.math.not_equal(
                self._entries_priority.lookup(str(content_ref)),
                self._entries_priority._default_value,
            )
            or all(
                tf.math.not_equal(
                    self._entries_content.lookup(str(priority)),
                    self._entries_content._default_value,
                )
            ),
            tf.bool,
        )
        
        if condt:
            tf.print(condt)
            update_item(content_ref, priority)
        else:
            create_item(content_ref, priority)

        return self.maxlen

    def pop(self):
        """
        Pops item with highest priority
        """

        if tf.math.not_equal(self.size(), tf.constant(0, tf.int32)):
            prev_priority = heapq.heappop(self._queue)
            prev_priority = str(prev_priority)

            content_ref = self._entries_content.lookup(prev_priority)
            content = self._get_unstacked_tensors(content_ref)

            content_ref = str(content_ref)
            self._entries_priority.erase(tf.expand_dims(content_ref, -1))
            self._entries_content.erase(tf.expand_dims(prev_priority, -1))

            content = list(
                map(
                    lambda c: tf.reshape(tf.cast(c[0], c[1]), c[2]),
                    zip(content, self.types, self.shapes),
                )
            )
            return content
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
