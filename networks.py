import tensorflow as tf
import innerlayers as il
from outernetwork import OuterNetwork

class OuterConvNetwork(OuterNetwork):
    def __init__(self, inner_variables, num_inner_loops):
        super().__init__(inner_variables=inner_variables, num_inner_loops=num_inner_loops)

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="VALID", activation="relu")
        self.conv11 = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding="VALID", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="VALID", activation="relu")
        self.conv21 = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding="VALID", activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="VALID", activation="relu")
        self.conv31 = tf.keras.layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding="VALID", activation="relu")

        self.dense1 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense2 = tf.keras.layers.Dense(self.output_size)

    def calculate_output(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Merge channels and inner-batch
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))
        shape = inputs.shape.as_list()
        inputs = tf.reshape(inputs, (batch_size, shape[1], shape[2], int(shape[3]) * int(shape[4])))
        weights = self.conv1(inputs)
        weights = self.conv11(weights)
        weights = self.conv2(weights)
        weights = self.conv21(weights)
        weights = self.conv3(weights)
        weights = self.conv31(weights)
        weights = tf.reshape(weights, (batch_size, tf.reduce_prod(weights.shape[1:])))
        weights = self.dense1(weights)
        weights = self.dense2(weights)
        self.output = weights

class InnerVAEEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.layers = []

        self.down_convs = tf.keras.models.Sequential([
            il.InnerConv2D(8, 2, (2, 2)),
            tf.keras.layers.ReLU(),
            il.InnerConv2D(8, 2, (2, 2)),
            #tf.keras.layers.ReLU(),
            #il.InnerConv2D(2*8, 3, (2, 2)),
        ])

        self.layers.append(self.down_convs)

    def call(self, inputs):
        output = self.down_convs(inputs)
        half_output_filters = output.shape[-1] // 2
        mean = output[:, :, :, :, :half_output_filters]
        logvar = output[:, :, :, :, half_output_filters:]
        return mean, logvar

class InnerVAEDecoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.layers = []

        self.up_convs = tf.keras.models.Sequential([
            il.InnerConv2DTranspose(8, 2, (2, 2)),
            tf.keras.layers.ReLU(),
            il.InnerConv2DTranspose(1, 2, (2, 2)),
            #tf.keras.layers.ReLU(),
            #il.InnerConv2DTranspose(1, 3, (2, 2)),
            #tf.keras.layers.Activation("sigmoid"),
        ])

        self.layers.append(self.up_convs)

    def call(self, latents):
        output = self.up_convs(latents)
        return output

class InnerVAE(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.encoder = InnerVAEEncoder()
        self.decoder = InnerVAEDecoder()
        self.layers = [self.encoder, self.decoder]

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, latents):
        return self.decoder(latents)

    def sample_normal(self, mean, logvar):
        return mean + tf.math.exp(0.5 * logvar) * tf.random.normal(logvar.shape)

    def get_loss(self, inputs):
        mean, logvar = self.encode(inputs)
        latents = self.sample_normal(mean, logvar)
        reconstr = self.decode(latents)
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=reconstr))
        kld = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=[1, 2, 3, 4])
        return kld + bce

    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        latents = self.sample_normal(mean, logvar)
        return tf.nn.sigmoid(self.decode(latents))