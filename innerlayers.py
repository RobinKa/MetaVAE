import tensorflow as tf

def get_inner_variables(layer, match_fn=None):
    inner_variables = []
    if isinstance(layer, InnerLayer):
        for inner_variable in layer.inner_variables.values():
            if match_fn is None or match_fn(inner_variable):
                inner_variables.append(inner_variable)

    if hasattr(layer, "layers"):
        for child_layer in layer.layers:
            inner_variables += get_inner_variables(child_layer, match_fn)

    return list(set(inner_variables))

def get_trainable_inner_variables(layer):
    return get_inner_variables(layer, match_fn=lambda inner_var: not inner_var.per_step)

def warmup_inner_layer(layer, input_shape):
    outer_batch_size = 1
    inner_batch_size = 1
    dummy_input = tf.placeholder(tf.float32, (outer_batch_size, inner_batch_size, *input_shape))
    layer(dummy_input)

class InnerVariable:
    counter = 0

    def __init__(self, shape, name=None, dtype=tf.float32, per_step=False):
        self.getter = lambda variable, batch_index: tf.placeholder(dtype, shape)
        self.name = name
        self.shape = shape
        self.per_step = per_step

        if self.name is None:
            self.name = "InnerVariable_%d" % InnerVariable.counter
            InnerVariable.counter += 1

    def get(self, batch_index):
        variable = self.getter(self, batch_index)
        assert variable is not None
        return variable

class InnerLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.inner_variables = {}

    def create_inner_variable(self, name, shape, dtype=tf.float32, per_step=False):
        if name in self.inner_variables:
            raise Exception("Tried to create inner variable with existing name")
        self.inner_variables[name] = InnerVariable(shape=shape, dtype=dtype, per_step=per_step)
        return self.inner_variables[name]

    def call(self, inputs):
        outer_batch_size = inputs.shape[0]
        results = []
        for batch_index in range(outer_batch_size):
            results.append(self.call_single(inputs[batch_index], batch_index))
        return tf.stack(results)

    def call_single(self, inputs, batch_index):
        pass

class InnerConv2D(InnerLayer):
    def __init__(self, filters, kernel_size, strides=(1, 1), use_bias=True, padding="VALID"):
        super().__init__()
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides = strides if len(strides) == 4 else (1, *strides, 1)
        self.use_bias = use_bias
        self.padding = padding

    def build(self, input_shape):
        self.kernel = self.create_inner_variable("kernel", (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters))
        if self.use_bias:
            self.bias = self.create_inner_variable("bias", (self.filters,))

    def call_single(self, inputs, batch_index):
        kernel = self.kernel.get(batch_index)
        output = tf.nn.conv2d(inputs, kernel, strides=self.strides, padding=self.padding)
        if self.use_bias:
            bias = self.bias.get(batch_index)
            output = tf.nn.bias_add(output, bias)
        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        # TODO: Correct padding values
        padding = (0, 0) if self.padding == "VALID" else (1000, 1000)

        output_shape = list(input_shape)
        output_shape[-3] = (input_shape[-3] - self.kernel_size[0] + 2 * padding[0]) // self.strides[1] + 1
        output_shape[-2] = (input_shape[-2] - self.kernel_size[1] + 2 * padding[1]) // self.strides[2] + 1
        output_shape[-1] = self.filters

        return tuple(output_shape)

class InnerConv2DTranspose(InnerLayer):
    def __init__(self, filters, kernel_size, strides=(1, 1), use_bias=True, padding="VALID"):
        super().__init__()
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.strides = strides if len(strides) == 4 else (1, *strides, 1)
        self.use_bias = use_bias
        self.padding = padding

    def build(self, input_shape):
        self.kernel = self.create_inner_variable("kernel", (self.kernel_size[0], self.kernel_size[1], self.filters, input_shape[-1]))
        if self.use_bias:
            self.bias = self.create_inner_variable("bias", (self.filters,))

    def call_single(self, inputs, batch_index):
        output_shape = self.compute_output_shape(inputs.shape)

        kernel = self.kernel.get(batch_index)
        output = tf.nn.conv2d_transpose(inputs, kernel, output_shape=output_shape, strides=self.strides, padding=self.padding)
        if self.use_bias:
            bias = self.bias.get(batch_index)
            output = tf.nn.bias_add(output, bias)

        return output

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        # TODO: Correct padding values
        padding = (0, 0) if self.padding == "VALID" else (1000, 1000)

        output_shape = list(input_shape)
        output_shape[-3] = (input_shape[-3] - 1) * self.strides[1] + self.kernel_size[0] - 2 * padding[0]
        output_shape[-2] = (input_shape[-2] - 1) * self.strides[2] + self.kernel_size[1] - 2 * padding[1]
        output_shape[-1] = self.filters
        
        return tuple(output_shape)

class InnerNormalization(InnerLayer):
    def __init__(self, per_step=True):
        super().__init__()
        self.per_step = per_step
        self.stored_mean = 0
        self.stored_var = 1

    def build(self, input_shape):
        self.std = self.create_inner_variable("std", (1,), per_step=self.per_step)
        self.mean = self.create_inner_variable("mean", (1,), per_step=self.per_step)

    def call_single(self, inputs, batch_index):
        std = self.std.get(batch_index)
        mean = self.mean.get(batch_index)
        output = std * inputs + mean
        return output

    def call(self, inputs):
        # Normalize to N(0, 1) over inner-batch axis together.
        # Then do the single-call normalization since every
        # inner batch has its own mean and std
        if inputs.shape[1] == 5:
            self.stored_mean, self.stored_var = tf.nn.moments(inputs, axes=[1], keep_dims=True)
        inputs = (inputs - self.stored_mean) / tf.sqrt(self.stored_var + 1e-6)
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape