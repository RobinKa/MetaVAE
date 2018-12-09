import tensorflow as tf
import numpy as np
import innerlayers as il
from networks import OuterConvNetwork, InnerVAE

class MetaVAE:
    def __init__(self, num_inner_loops=5):
        self.num_inner_loops = num_inner_loops

        # Zero variables
        self.input_shape = None
        self.inner_vae = None
        self.outer_network = None
        self.inner_vars = None
        self.trainable_inner_vars = None

    def _build(self, input_shape):
        assert self.input_shape is None or self.input_shape == input_shape

        if self.input_shape == input_shape:
            return

        self.input_shape = input_shape

        self.inner_vae = InnerVAE()

        # Warmup the inner network so the layers are built and 
        # we can collect the inner variables needed for the
        # outer network.
        il.warmup_inner_layer(self.inner_vae, input_shape)

        self.inner_vars = il.get_inner_variables(self.inner_vae)
        print("Found inner vars:", self.inner_vars)

        # Collect mutable inner variables from inner network.
        # If the outer network outputs the inner variable per-step
        # then we can not mutate it.
        self.trainable_inner_vars = il.get_trainable_inner_variables(self.inner_vae)

        self.outer_network = OuterConvNetwork(self.inner_vars, num_inner_loops=self.num_inner_loops)


    def get_loss(self, train_inputs, test_inputs):
        assert train_inputs.shape[2:] == test_inputs.shape[2:]
        self._build(train_inputs.shape[2:])

        def _image_summary(name, images):
            tf.summary.image(name, tf.cast(255 * images[:, 0], tf.uint8))

        def _avg_scalar_summary(name, values):
            tf.summary.scalar(name, tf.reduce_mean(values))

        # Keep initial and step test losses and average them
        # in the end.
        test_losses = []

        # Calculate initial weights using outer network
        self.outer_network.calculate_output(train_inputs)

        # Collect initial values for mutable inner variables from
        # outer network.
        mutable_inner_vars = {
            inner_var: self.outer_network.get_inner_variable(inner_var, step=0)
            for inner_var in self.trainable_inner_vars
        }

        # Create the inner variable getter method.
        # Must make sure that the step variable exists (created in loop below).
        def _get_inner_var(inner_var, batch_index):
            if inner_var.per_step:
                return self.outer_network.get_inner_variable(inner_var, step)[batch_index]
            else:
                return mutable_inner_vars[inner_var][batch_index]

        # Assign getters to the function above for the inner variables
        for inner_var in self.inner_vars:
            inner_var.getter = _get_inner_var

        # Calculate initial test loss
        initial_test_loss = self.inner_vae.get_loss(test_inputs)
        test_losses.append(initial_test_loss)
        _avg_scalar_summary("test_loss_initial", initial_test_loss)
        _image_summary("test_input", test_inputs)

        for step in range(self.num_inner_loops):
            # Calculate train loss for this step
            step_train_loss = self.inner_vae.get_loss(train_inputs)

            # Mutable inner variable gradient update using train loss for this step
            mutable_inner_vars_keys, mutable_inner_vars_values = list(mutable_inner_vars.keys()), list(mutable_inner_vars.values())
            mutable_inner_vars_grads = tf.gradients(step_train_loss, mutable_inner_vars_values)
            for inner_var, weights, grads in zip(mutable_inner_vars_keys, mutable_inner_vars_values, mutable_inner_vars_grads):
                mutable_inner_vars[inner_var] = weights - 0.01 * grads
        
            # Calculate test loss for this step
            step_test_loss = self.inner_vae.get_loss(test_inputs)
            test_losses.append(step_test_loss)
            _avg_scalar_summary("test_loss_step_%d" % step, step_test_loss)

        test_loss_total = tf.reduce_mean(test_losses)
        _avg_scalar_summary("test_loss_total", test_loss_total)
        return test_loss_total
