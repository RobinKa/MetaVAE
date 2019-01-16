import tensorflow as tf
import inner as il
from outer import OuterNetwork

class OuterConstantNetwork(OuterNetwork):
    def __init__(self, inner_variables, num_inner_loops, fixed_lr=None):
        super().__init__(inner_variables=inner_variables, num_inner_loops=num_inner_loops, fixed_lr=fixed_lr)
        self.constant_init = tf.get_variable("inner_init", (self.output_size,), dtype=tf.float32, trainable=True)

    def calculate_output(self, inputs):
        batch_size = tf.shape(inputs)[0]
        self.output = tf.tile(tf.expand_dims(self.constant_init, 0), (batch_size, 1))

class OuterSeperatedConstantNetwork(OuterNetwork):
    def __init__(self, inner_variables, num_inner_loops, fixed_lr=None):
        super().__init__(inner_variables=inner_variables, num_inner_loops=num_inner_loops, fixed_lr=fixed_lr)
        self.inner_var_constants = None
        self.learning_rates = None
                
    def get_inner_variable(self, inner_variable, step):
        """
        Gets the values for the inner variable at the specified step.
        Returns one variable for every outer batch.
        [OuterBatchSize, *VariableShape]
        """
        assert step >= 0 and step <= self.num_inner_loops and (inner_variable.per_step or step == 0)
        return self.inner_var_constants[inner_variable][step]

    def get_learning_rate(self, inner_variable, step):
        assert not inner_variable.per_step and step >= 0 and step < self.num_inner_loops
        if self.fixed_lr:
            return self.fixed_lr
        else:
            return self.learning_rates[inner_variable][step]

    def calculate_output(self, inputs):
        batch_size = tf.shape(inputs)[0]

        self.inner_var_constants = {}
        self.learning_rates = {}

        for inner_var in self.inner_variables:
            print(inner_var.name, inner_var.shape)
            self.inner_var_constants[inner_var] = []
            if inner_var.per_step:
                # Need 1 more step than inner loops (since after training we need another set of variables)
                for step in range(self.num_inner_loops + 1):
                    tf_var = tf.get_variable("inner_var_%s_step_%d" % (inner_var.name, step), inner_var.shape, dtype=inner_var.dtype, trainable=True, initializer=inner_var.initializer)
                    self.inner_var_constants[inner_var].append(tf_var)
            else:
                tf_var = tf.get_variable("inner_var_%s" % (inner_var.name), inner_var.shape, dtype=inner_var.dtype, trainable=True, initializer=inner_var.initializer)
                tf_var = tf.tile(tf.expand_dims(tf_var, 0), (batch_size, *([1] * len(inner_var.shape))))
                self.inner_var_constants[inner_var].append(tf_var)
                if self.fixed_lr is None:
                    self.learning_rates[inner_var] = []
                    for step in range(self.num_inner_loops):
                        lr_var = tf.get_variable("inner_var_%s_lr_step_%d" % (inner_var.name, step), (1,), dtype=tf.float32, trainable=True, initializer=tf.constant_initializer(0.3))
                        self.learning_rates[inner_var].append(lr_var)

class OuterLinearNetwork(OuterNetwork):
    def __init__(self, inner_variables, num_inner_loops):
        super().__init__(inner_variables=inner_variables, num_inner_loops=num_inner_loops)
        self.dense = tf.keras.layers.Dense(self.output_size)

    def calculate_output(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs = tf.reshape(inputs, (batch_size, -1))
        self.output = self.dense(inputs)

class OuterConvNetwork(OuterNetwork):
    def __init__(self, inner_variables, num_inner_loops):
        super().__init__(inner_variables=inner_variables, num_inner_loops=num_inner_loops)

        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", activation=tf.keras.layers.LeakyReLU(0.2))
        self.conv11 = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation=tf.keras.layers.LeakyReLU(0.2))
        self.bn1 = tf.keras.layers.BatchNormalization() #32
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", activation=tf.keras.layers.LeakyReLU(0.2))
        self.conv21 = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation=tf.keras.layers.LeakyReLU(0.2))
        self.bn2 = tf.keras.layers.BatchNormalization() #16
        self.conv3 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", activation=tf.keras.layers.LeakyReLU(0.2))
        self.conv31 = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation=tf.keras.layers.LeakyReLU(0.2))
        self.bn3 = tf.keras.layers.BatchNormalization() #8
        self.conv4 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", activation=tf.keras.layers.LeakyReLU(0.2))
        self.conv41 = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation=tf.keras.layers.LeakyReLU(0.2))
        self.bn4 = tf.keras.layers.BatchNormalization() #4
        self.conv5 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", activation=tf.keras.layers.LeakyReLU(0.2))
        self.conv51 = tf.keras.layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding="SAME", activation=tf.keras.layers.LeakyReLU(0.2))
        self.bn5 = tf.keras.layers.BatchNormalization() #2
        self.conv6 = tf.keras.layers.Conv2D(32, kernel_size=(2, 2), strides=(1, 1), padding="VALID", activation=tf.keras.layers.LeakyReLU(0.2))
        #1

        #self.dense1 = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(0.2))
        self.dense2 = tf.keras.layers.Dense(self.output_size)

    def calculate_output(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Merge channels and inner-batch
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))
        shape = inputs.shape.as_list()
        inputs = tf.reshape(inputs, (batch_size, shape[1], shape[2], int(shape[3]) * int(shape[4])))
        print("Shape (expected 64x64):", inputs.shape)
        weights = self.conv1(inputs)
        weights = self.conv11(weights)
        weights = self.bn1(weights)
        print("Shape (expected 32x32):", weights.shape)
        weights = self.conv2(weights)
        weights = self.conv21(weights)
        weights = self.bn2(weights)
        print("Shape (expected 16x16):", weights.shape)
        weights = self.conv3(weights)
        weights = self.conv31(weights)
        weights = self.bn3(weights)
        print("Shape (expected 8x8):", weights.shape)
        weights = self.conv4(weights)
        weights = self.conv41(weights)
        weights = self.bn4(weights)
        print("Shape (expected 4x4):", weights.shape)
        weights = self.conv5(weights)
        weights = self.conv51(weights)
        weights = self.bn5(weights)
        print("Shape (expected 2x2):", weights.shape)
        weights = self.conv6(weights)
        print("Shape (expected 1x1):", weights.shape)
        weights = tf.reshape(weights, (batch_size, tf.reduce_prod(weights.shape[1:])))
        #weights = self.dense1(weights)
        weights = self.dense2(weights)

        def _make_sane(x):
            if isinstance(x, tf.Dimension):
                x = x.value
            return int(x)

        fixed_bias = []
        for inner_var in self.inner_variables:
            initializer = inner_var.initializer
            shape = tuple(_make_sane(x) for x in inner_var.shape)

            if inner_var.per_step:
                # Need 1 more step than inner loops (since after training we need another set of variables)
                for step in range(self.num_inner_loops + 1):
                    var = initializer(shape=shape, dtype=inner_var.dtype)
                    var = tf.reshape(var, (-1,))
                    fixed_bias.append(var)
            else:
                var = initializer(shape=shape, dtype=inner_var.dtype)
                var = tf.reshape(var, (-1,))
                fixed_bias.append(var)
                if self.fixed_lr is None:
                    fixed_bias.append(tf.constant(0.01, dtype=tf.float32, shape=(self.num_inner_loops,)))

        fixed_bias = tf.concat(fixed_bias, axis=0)
        fixed_bias = tf.expand_dims(fixed_bias, axis=0)

        self.output = weights + fixed_bias

class InnerVAEEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.down_convs = tf.keras.models.Sequential()
        size = 32
        while size > 2:
            #self.down_convs.add(il.InnerMemorization())
            #self.down_convs.add(il.InnerConv2D(32, 3, (1, 1), padding="SAME", use_bias=False))
            #self.down_convs.add(tf.keras.layers.LeakyReLU(0.2))

            self.down_convs.add(il.InnerConv2D(32, 3, (2, 2), padding="SAME", use_bias=False))
            #self.down_convs.add(il.InnerNormalization())
            self.down_convs.add(tf.keras.layers.LeakyReLU(0.2))
            size //= 2

        assert size == 2

        #self.down_convs.add(il.InnerConv2D(128, 3, (1, 1), padding="SAME", use_bias=True))
        #self.down_convs.add(il.InnerConv2D(64, 2, (1, 1), padding="VALID", use_bias=True))
        #self.down_convs.add(il.InnerNormalization())
        #self.down_convs.add(tf.keras.layers.LeakyReLU(0.2))

        self.down_convs.add(il.InnerReshape((128,)))
        self.down_convs.add(il.InnerDense(256))

        self.layers = [self.down_convs]

    def call(self, inputs):
        output = self.down_convs(inputs)
        half_output = output.shape[-1] // 2
        mean = output[:, :, :half_output]
        logvar = output[:, :, half_output:]
        return mean, logvar

class InnerVAEDecoder(tf.keras.layers.Layer):
    def __init__(self, output_channels):
        super().__init__()

        self.layers = []

        self.up_convs = tf.keras.models.Sequential()
        
        self.up_convs.add(il.InnerDense(128, use_bias=False))
        self.up_convs.add(il.InnerReshape((2, 2, 32)))
        self.up_convs.add(il.InnerNormalization())
        self.up_convs.add(tf.keras.layers.LeakyReLU(0.2))

        size = 2
        while size < 32:
            #self.up_convs.add(il.InnerMemorization())
            #self.up_convs.add(il.InnerConv2D(32, min(size, 3), (1, 1), padding="SAME", use_bias=False))
            #self.up_convs.add(tf.keras.layers.LeakyReLU(0.2))

            self.up_convs.add(il.InnerConv2D(32, min(size, 3), (1, 1), padding="SAME", use_bias=False))
            self.up_convs.add(il.InnerNormalization())
            self.up_convs.add(il.InnerResize((size*2, size*2)))
            self.up_convs.add(tf.keras.layers.LeakyReLU(0.2))
            size *= 2

        # Final refinement
        self.up_convs.add(il.InnerConv2D(output_channels, 3, (1, 1), padding="SAME", use_bias=True))
        #self.up_convs.add(il.InnerMemorization())

        self.layers = [self.up_convs]

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
        #bce = tf.reduce_mean(tf.abs(inputs - tf.nn.sigmoid(reconstr)), axis=list(range(1, len(inputs.shape))))
        kld = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=list(range(1, len(mean.shape))))
        return {"loss": kld + bce, "latents": latents, "reconstruction": tf.nn.sigmoid(reconstr), "bce": bce, "kld": kld}

    def call(self, inputs):
        mean, logvar = self.encode(inputs)
        latents = self.sample_normal(mean, logvar)
        return tf.nn.sigmoid(self.decode(latents))