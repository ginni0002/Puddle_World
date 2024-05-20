import tensorflow as tf
from utils import ReplayBuffer


class QNetwork:

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        input_spec: tuple,
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
        self.model.compile(self.opt, self.loss, metrics=["accuracy"])

        self.data_spec = input_spec
        self.replay_buffer = ReplayBuffer(self.data_spec, buffer_length)

    
    def _build_model(self):

        inp = tf.keras.layers.Input(
            shape=(self.input_dims, 1),
            # batch_size=self.batch_size,
        )
        x = inp
        for dim in self.dims:
            x = tf.keras.layers.Dense(dim, activation=tf.nn.relu)(x)

        x = tf.keras.layers.Flatten(name="flatten")(x)
        out = tf.keras.layers.Dense(self.n_output)(x)
        return tf.keras.Model(inputs=inp, outputs=out)

    @tf.function(input_signature=(tf.TensorSpec([None, 1], dtype=tf.float32),))
    def predict(self, state):
        """
        Predict Q values of all actions given state
        """
        # convert state to batch
        # state = tf.expand_dims(state, 0)
        state = [tf.reshape(state, (1, self.input_dims))]
        return tf.squeeze(self.model(state))

    @tf.function
    def learn(self):
        """
        Update QNetwork parameters given replay mini-batch
        """
        sample = self.replay_buffer.get_random_samples()
        # compute Q-targets and current Q values, get loss from TD error
        # take gradient over batch
        sample = tf.squeeze(tf.stop_gradient(sample))
        s, a, r, s_ = tf.unstack(sample, axis=1)
        a, _ = tf.unstack(tf.cast(a, tf.int32), axis=1)
        r, _ = tf.unstack(r, axis=1)
        r = tf.expand_dims(r, -1)
        with tf.GradientTape() as tape:
            q_target = r + self.gamma * tf.reduce_max(self.model(s_), 1, keepdims=True)

            action_values = self.model(s)
            q_current = tf.gather(action_values, a, axis=1, batch_dims=1)
            q_current = tf.reshape(q_current, q_target.shape)
            loss = self.loss(q_target, q_current)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss
