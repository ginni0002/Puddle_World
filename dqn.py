import tensorflow as tf
from collections import deque


class QNetwork:

    def __init__(
        self,
        input_dims: int,
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
                [input_dims, 1],
                tf.float32,
                "state",
            ),
            tf.TensorSpec([input_dims, 1], tf.float32, "action"),
            tf.TensorSpec([input_dims, 1], tf.float32, "reward"),
            tf.TensorSpec(
                [input_dims, 1],
                tf.float32,
                "next_state",
            ),
        )
        self.spec_shapes = [i.shape for i in self.data_spec]
        self.replay_buffer = deque(maxlen=buffer_length)

    def _build_model(self):

        inp = tf.keras.layers.Input(
            shape=self.input_dims,
            batch_size=self.batch_size,
        )
        x = inp
        for dim in self.dims:
            x = tf.keras.layers.Dense(dim, activation=tf.nn.relu)(x)

        out = tf.keras.layers.Dense(self.n_output)(x)
        return tf.keras.Model(inputs=[inp], outputs=[out])

    def collect_rollout(self, sars: tuple):
        """
        Collect samples for the replay buffer for batch updates
        """
        try:
            sars = tf.keras.utils.pad_sequences(sars, dtype="float32")
            sars = tf.expand_dims(sars, -1)
        except ValueError:
            print(f"Invalid (s, a, r, s') tuple: {sars}")
            print(f"Valid shapes: {self.spec_shapes}")
            raise

        self.replay_buffer.append(sars)

    @tf.function(input_signature=(tf.TensorSpec([None, 1], dtype=tf.float32),))
    def predict(self, state):
        """
        Predict Q values of all actions given state
        """
        # convert state to batch
        state = tf.expand_dims(state, 0)
        return self.model(state)

    @tf.function
    def learn(self):
        """
        Update QNetwork parameters given replay mini-batch
        """
        indices = tf.random.uniform(
            (self.batch_size,), 0, len(self.replay_buffer), dtype=tf.int32
        )
        sample = tf.gather(tf.identity(list(self.replay_buffer)), indices)
        # compute Q-targets and current Q values, get loss from TD error
        # take gradient over batch
        s, a, r, s_ = tf.unstack(tf.stop_gradient(sample), axis=1)
        a, _ = tf.unstack(tf.cast(a, tf.int32), axis=1)
        r, _ = tf.unstack(r, axis=1)
        with tf.GradientTape() as tape:
            q_target = r + self.gamma * tf.reduce_max(self.model(s_), 1, keepdims=True)

            action_values = self.model(s)
            q_current = tf.gather_nd(action_values, a, batch_dims=1)
            loss = self.loss(q_target, q_current)
            td_error = q_target - q_current
            
        grads = tape.gradient(loss, self.model.trainable_weights)
        grads = self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
        return grads, td_error
