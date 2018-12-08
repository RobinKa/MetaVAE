import tensorflow as tf
import numpy as np

def single_inner_conv2d(params):
    inputs, weights = params
    output = tf.nn.conv2d(inputs, weights, strides=(1, 2, 2, 1), padding="VALID")
    #print("conv2d output:", output)
    return output

def single_inner_conv2d_transpose(params):
    inputs, weights = params

    #pad = (1 + inputs.shape[1] % 2 , 1 + inputs.shape[2] % 2)
    #out_size = [weights.shape[0] + (inputs.shape[1] - 1) * 2, weights.shape[1] + (inputs.shape[2] - 1) * 2]
    #out_size = [inputs.shape[1] * 2, inputs.shape[2] * 2]
    out_dict = {2: 6, 6: 13, 13: 28}
    out_size = [out_dict[int(inputs.shape[1])], out_dict[int(inputs.shape[2])]]

    shape = tf.TensorShape([inputs.shape[0], *out_size, weights.shape[2]])
    #print("Out shape:", shape)
    output = tf.nn.conv2d_transpose(inputs, weights, output_shape=shape, strides=(1, 2, 2, 1), padding="VALID")
    #print("Output:", output)
    return output

def make_mapped_fn(fn, *params):
    outputs = []
    for i in range(32):
        pp = [p[i] for p in params]
        outputs.append(fn(pp))
    outputs = tf.stack(outputs)
    print("mapped output:", outputs)
    return outputs
    #return tf.map_fn(fn, params, parallel_iterations=10)

inner_conv2d = lambda inputs, weights: make_mapped_fn(single_inner_conv2d, inputs, weights)
inner_conv2d_transpose = lambda inputs, weights: make_mapped_fn(single_inner_conv2d_transpose, inputs, weights)

class InnerVAE:
    def encode(self, inputs, layer_weights):
        encoded = inputs
        for weights in layer_weights:
            encoded = inner_conv2d(encoded, weights)
            if weights is not layer_weights[-1]:
                encoded = tf.nn.leaky_relu(encoded)

        encoded_mean = encoded[:, :, :, :, :layer_weights[-1].shape[-1]//2]
        encoded_logvar = encoded[:, :, :, :, layer_weights[-1].shape[-1]//2:]
        print("Encoded mean:", encoded_mean)
        print("Encoded logvar:", encoded_logvar)
        return encoded_mean, encoded_logvar

    def sample_latents(self, mean, logvar):
        return mean + tf.math.exp(0.5 * logvar) * tf.random.normal(logvar.shape)

    def decode(self, latents, layer_weights):
        decoded = latents

        for weights in layer_weights:
            decoded = inner_conv2d_transpose(decoded, weights)
            if weights is layer_weights[-1]:
                decoded = tf.nn.sigmoid(decoded + 2.44) # HACK: hardcoded offset since we don't have biases atm
            else:
                decoded = tf.nn.leaky_relu(decoded)
        return decoded

class MetaNetwork:
    def __init__(self, weight_shapes, num_inner_loops):
        self.weight_shapes = weight_shapes
        self.num_inner_loops = num_inner_loops
        num_outputs = sum([np.prod(shape) for shape in weight_shapes]) + len(weight_shapes) * num_inner_loops

        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding="VALID", activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding="VALID", activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding="VALID", activation="relu")

        self.dense1 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense2 = tf.keras.layers.Dense(num_outputs)

    def get_weights(self, inputs):
        print("Get weights input:", inputs)
        batch_size = tf.shape(inputs)[0]
        # Merge channels and inner-batch
        inputs = tf.transpose(inputs, (0, 2, 3, 4, 1))
        shape = inputs.shape.as_list()
        inputs = tf.reshape(inputs, (batch_size, shape[1], shape[2], int(shape[3]) * int(shape[4])))
        print("Preconv:", inputs)
        weights = self.conv1(inputs)
        weights = self.conv2(weights)
        weights = self.conv3(weights)
        print("After conv:", weights)
        weights = tf.reshape(weights, (batch_size, tf.reduce_prod(weights.shape[1:])))
        print("After reshape:", weights)
        weights = self.dense1(weights)
        weights = self.dense2(weights)

        layer_weights = []
        index = 0
        for weight_shape in self.weight_shapes:
            layer_size = np.prod(weight_shape)
            layer_weights.append(tf.reshape(weights[:, index:index+layer_size], [-1, *weight_shape]))
            index += layer_size

        learning_rates = []
        for step in range(self.num_inner_loops):
            learning_rates.append(weights[:, index:index+len(self.weight_shapes)])
            index += len(self.weight_shapes)

        print("Layer weights:", layer_weights)
        print("Layer learning rates:", learning_rates)

        return layer_weights, learning_rates

class MetaVAE:
    def __init__(self, num_inner_loops=5):
        k = 3
        f = 8
        #self.encoder_shapes = [[k, k, 1, f], [k, k, f, f*2*2]]
        #self.decoder_shapes = [[k, k, f, f*2], [k, k, 1, f]]
        self.encoder_shapes = [[k, k, 1, f], [k, k, f, f*2], [k, k, f*2, f*4*2]]
        self.decoder_shapes = [[k, k, f*2, f*4], [k, k, f, f*2], [k, k, 1, f]]
        self.meta_network = MetaNetwork(self.encoder_shapes + self.decoder_shapes, num_inner_loops)
        self.inner_network = InnerVAE()
        self.num_inner_loops = num_inner_loops

    def get_loss(self, inputs):
        return self._get_meta_loss(inputs[:, :5], inputs[:, 5:])

    def _get_meta_loss(self, train_inputs, test_inputs):
        """inputs [MetaBatch, InnerBatch, W, H, C]"""

        print("Train inputs:", train_inputs)
        print("Test inputs:", test_inputs)

        def _get_vae_loss(inputs, reconstr, latents_mean, latents_logvar):
            #mse = 0.5 * tf.reduce_mean(tf.square(inputs - reconstr), axis=[1, 2, 3, 4])
            bce = -tf.reduce_mean(inputs * tf.log(1e-7 + reconstr) + (1 - inputs) * tf.log(1e-7 + 1 - reconstr), axis=[1, 2, 3, 4])
            #mae = tf.reduce_mean(tf.abs(inputs - reconstr), axis=[1, 2, 3, 4])
            kld = -0.5 * tf.reduce_mean(1 + latents_logvar - tf.square(latents_mean) - tf.exp(latents_logvar), axis=[1, 2, 3, 4])
            return kld + bce, kld, bce#mae

        def _image_summary(name, images):
            tf.summary.image(name, tf.cast(255 * images[:, 0], tf.uint8))

        def _avg_scalar_summary(name, values):
            tf.summary.scalar(name, tf.reduce_mean(values))

        # list([MetaBatch, *LayerShape])
        layer_weights, learning_rates = self.meta_network.get_weights(train_inputs)
        get_enc_weights = lambda: layer_weights[:len(self.encoder_shapes)]
        get_dec_weights = lambda: layer_weights[len(self.encoder_shapes):len(self.encoder_shapes)+len(self.decoder_shapes)]

        #learning_rates = tf.Print(learning_rates, learning_rates, message="Per-step per-layer learning rates", summarize=5*4)

        # Calculate initial test loss
        enc_mean, enc_logvar = self.inner_network.encode(test_inputs, get_enc_weights())
        lat = self.inner_network.sample_latents(enc_mean, enc_logvar)
        dec = self.inner_network.decode(lat, get_dec_weights())
        initial_test_loss, kld, bce = _get_vae_loss(test_inputs, dec, enc_mean, enc_logvar)
        test_loss = [initial_test_loss]

        _image_summary("test_input", test_inputs)
        _image_summary("test_reconstructed_initial", dec)

        _avg_scalar_summary("test_loss_initial", initial_test_loss)
        _avg_scalar_summary("test_kld_initial", kld)
        _avg_scalar_summary("test_bce_initial", bce)

        for step in range(self.num_inner_loops):
            # [MetaBatch, InnerBatch, *LatentsShape]
            enc_mean, enc_logvar = self.inner_network.encode(train_inputs, get_enc_weights())
            print("Train enc:", enc_mean, enc_logvar)

            # [MetaBatch, InnerBatch, W, H, C]
            lat = self.inner_network.sample_latents(enc_mean, enc_logvar)
            dec = self.inner_network.decode(lat, get_dec_weights())
            print("Train dec:", dec)

            # [MetaBatch]
            step_loss, kld, bce = _get_vae_loss(train_inputs, dec, enc_mean, enc_logvar)
            print("Step train loss:", step_loss)

            # [Layers, MetaBatch, *LayerShape]
            layer_grads = tf.gradients(step_loss, layer_weights, stop_gradients=layer_weights)
            print("Batched layer grads len:", len(layer_grads))
            print("Layer grads first element len:", layer_grads[0].shape)
            
            for i, layer_weight in enumerate(layer_weights):
                # [*LayerShape] = d[MetaBatch] / d[MetaBatch, *LayerShape]
                print("Layer weight:", layer_weight)
                print("grads:", layer_grads[i])
                layer_lr = tf.reshape(learning_rates[step][:, i], [-1] + [1] * (len(layer_weight.shape) - 1))
                layer_weights[i] = layer_weight - layer_lr * layer_grads[i]
        
            # [MetaBatch, InnerBatch, *LatentsShape]
            enc_mean, enc_logvar = self.inner_network.encode(test_inputs, get_enc_weights())
            print("Test enc:", enc_mean, enc_logvar)
            
            # [MetaBatch, InnerBatch, W, H, C]
            lat = self.inner_network.sample_latents(enc_mean, enc_logvar)
            dec = self.inner_network.decode(lat, get_dec_weights())
            print("Test dec:", dec)

            _image_summary("test_reconstructed_step_%d" % step, dec)

            # [MetaBatch]
            step_test_loss, kld, bce = _get_vae_loss(test_inputs, dec, enc_mean, enc_logvar)
            print("Step test loss:", step_test_loss)
            _avg_scalar_summary("test_loss_step_%d" % step, step_test_loss)
            _avg_scalar_summary("test_kld_step_%d" % step, kld)
            _avg_scalar_summary("test_bce_step_%d" % step, bce)
            test_loss.append(step_test_loss)

        # Make extra summaries
        lat = self.inner_network.sample_latents(tf.zeros_like(enc_mean), tf.ones_like(enc_logvar))
        random_samples = self.inner_network.decode(lat, get_dec_weights())
        _image_summary("random_samples", random_samples)

        #test_loss = 0.04 * test_loss[0] + 0.06 * test_loss[1] + 0.1 * test_loss[2] + 0.16 * test_loss[3] + 0.2 * test_loss[4] + 0.44 * test_loss[5] 
        _avg_scalar_summary("test_loss_total", test_loss)
        return tf.reduce_mean(test_loss)
