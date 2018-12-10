import tensorflow as tf
import innerlayers as il
from outernetwork import OuterNetwork

class OuterConstantNetwork(OuterNetwork):
    def __init__(self, inner_variables, num_inner_loops, fixed_lr=None):
        super().__init__(inner_variables=inner_variables, num_inner_loops=num_inner_loops, fixed_lr=fixed_lr)
        self.constant_init = tf.get_variable("inner_init", (self.output_size,), dtype=tf.float32, trainable=True)

    def calculate_output(self, inputs):
        batch_size = tf.shape(inputs)[0]
        self.output = tf.tile(tf.expand_dims(self.constant_init, 0), (batch_size, 1))

class OuterConvNetwork(OuterNetwork):
    def __init__(self, inner_variables, num_inner_loops):
        super().__init__(inner_variables=inner_variables, num_inner_loops=num_inner_loops)

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="VALID", activation=tf.keras.layers.LeakyReLU(0.2))
        self.conv11 = tf.keras.layers.Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding="VALID", activation=tf.keras.layers.LeakyReLU(0.2))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="VALID", activation=tf.keras.layers.LeakyReLU(0.2))
        self.conv21 = tf.keras.layers.Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding="VALID", activation=tf.keras.layers.LeakyReLU(0.2))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="VALID", activation=tf.keras.layers.LeakyReLU(0.2))
        self.conv31 = tf.keras.layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding="VALID", activation=tf.keras.layers.LeakyReLU(0.2))
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.dense1 = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(0.2))
        self.dense2 = tf.keras.layers.Dense(self.output_size)

    def calculate_output(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Merge channels and inner-batch
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))
        shape = inputs.shape.as_list()
        inputs = tf.reshape(inputs, (batch_size, shape[1], shape[2], int(shape[3]) * int(shape[4])))
        weights = self.conv1(inputs)
        weights = self.conv11(weights)
        weights = self.bn1(weights)
        weights = self.conv2(weights)
        weights = self.conv21(weights)
        weights = self.bn2(weights)
        weights = self.conv3(weights)
        weights = self.conv31(weights)
        weights  = self.bn3(weights)
        weights = tf.reshape(weights, (batch_size, tf.reduce_prod(weights.shape[1:])))
        weights = self.dense1(weights)
        weights = self.dense2(weights)
        self.output = weights

class InnerVAEEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.layers = []

        self.down_convs = tf.keras.models.Sequential([
            il.InnerConv2D(16, 4, (1, 1)),
            tf.keras.layers.LeakyReLU(0.2),

            il.InnerNormalization(),
            il.InnerConv2D(16, 4, (1, 1)),
            tf.keras.layers.LeakyReLU(0.2),

            il.InnerNormalization(),
            il.InnerConv2D(16, 4, (2, 2)),
            tf.keras.layers.LeakyReLU(0.2),

            il.InnerNormalization(),
            il.InnerConv2D(16, 3, (1, 1)),
            tf.keras.layers.LeakyReLU(0.2),

            il.InnerNormalization(),
            il.InnerConv2D(8*2, 4, (2, 2)),
            #tf.keras.layers.LeakyReLU(0.2),

            #il.InnerNormalization(),
            #il.InnerConv2D(16*2, 4, (1, 1)),
        ])

        self.layers.append(self.down_convs)

    def call(self, inputs):
        output = self.down_convs(inputs)
        half_output_filters = output.shape[-1] // 2
        mean = output[:, :, :, :, :half_output_filters]
        logvar = output[:, :, :, :, half_output_filters:]
        return mean, logvar

class InnerVAEDecoder(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()

        self.layers = []

        self.up_convs = tf.keras.models.Sequential([
            #il.InnerConv2DTranspose(4, 4, (2, 2), use_bias=False),
            #tf.keras.layers.LeakyReLU(0.2),

            #il.InnerNormalization(),
            il.InnerConv2DTranspose(16, 4, (2, 2), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),

            #il.InnerNormalization(),
            il.InnerConv2DTranspose(16, 4, (2, 2), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),

            il.InnerNormalization(),
            il.InnerConv2DTranspose(8, 4, (1, 1), use_bias=True),
            tf.keras.layers.LeakyReLU(0.2),

            #il.InnerNormalization(),
            il.InnerConv2DTranspose(output_channels, 4, (1, 1)),
        ])

        self.layers.append(self.up_convs)

    def call(self, latents):
        output = self.up_convs(latents)
        return output

class InnerVAE(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()

        self.encoder = InnerVAEEncoder()
        self.decoder = InnerVAEDecoder(output_channels=output_channels)
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
        bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs, logits=reconstr), axis=[1, 2, 3, 4])
        kld = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=[1, 2, 3, 4])
        return {"loss": kld + bce, "latents": latents, "reconstruction": tf.nn.sigmoid(reconstr), "bce": bce, "kld": kld}

    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        latents = self.sample_normal(mean, logvar)
        return tf.nn.sigmoid(self.decode(latents))