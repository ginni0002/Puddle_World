import heapq
import time
import io

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt


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
            and isinstance(item[0], float | np.float32 | int)
        ), "Item should be a tuple of length 2, (priority: float | int, content: any)"

        # since heapq implements min-heap, invert priority value
        priority, content = item
        priority = -abs(priority)
        if content in self._entries:
            if priority < self._entries[content]:
                t1 = time.time()
                item = tuple([priority, content])
                arr = np.array(self._queue, dtype=object)[:, 1]
                arr = np.array([hash(i) for i in arr])
                idx = int(np.squeeze(np.where(arr == hash(content))))
                self._queue[idx] = item
                self._entries[content] = priority
                print(time.time() - t1)
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
