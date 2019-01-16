import tensorflow as tf
import numpy as np
import inner as il
from outer import OuterNetwork
from networks import OuterConstantNetwork, OuterLinearNetwork, OuterConvNetwork, OuterSeperatedConstantNetwork

class InnerConvClassifier(tf.keras.layers.Layer):
    def __init__(self, num_ways, num_convs=4, num_filters=64):
        super().__init__()

        self.num_ways = num_ways

        self.modules = tf.keras.models.Sequential()
        for _ in range(num_convs):
            self.modules.add(il.InnerConv2D(num_filters, 3, strides=(2, 2), padding="SAME", use_bias=False))
            self.modules.add(il.InnerNormalization())
            self.modules.add(tf.keras.layers.LeakyReLU(0.2))
        
        self.modules.add(il.InnerFlatten())
        self.modules.add(il.InnerDense(num_ways))

        self.layers = [self.modules]

    def get_loss(self, inputs, labels):
        logits = self.modules(inputs)

        # Logits: [OuterBatch, NumWays]
        # Labels: [OuterBatch] with values between 0 and NumWays-1 (incl.)

        gt = tf.one_hot(labels, self.num_ways)
        #cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits), axis=[1])
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt, logits=logits), axis=-1)
        print("Logits:", logits)
        print("Labels:", labels)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, tf.argmax(logits, axis=-1, output_type=tf.int32)), tf.float32), axis=[1])
        print("Accuracy:", accuracy)
        return cross_entropy, accuracy

    def call(self, inputs):
        return tf.nn.softmax(self.modules(inputs))

class MetaConvClassifier:
    def __init__(self, num_ways, num_inner_loops=5, first_order=False, adjust_loss=False):
        il.InnerVariable.counter = 0

        self.num_inner_loops = num_inner_loops
        self.first_order = first_order
        self.num_ways = num_ways
        self.adjust_loss = adjust_loss

        # Zero variables
        self.input_shape = None
        self.inner_classifier = None
        self.outer_network = None
        self.inner_vars = None
        self.trainable_inner_vars = None

    def _build(self, input_shape):
        assert self.input_shape is None or self.input_shape == input_shape

        if self.input_shape == input_shape:
            return

        self.input_shape = input_shape

        self.inner_classifier = InnerConvClassifier(self.num_ways)

        # Warmup the inner network so the layers are built and 
        # we can collect the inner variables needed for the
        # outer network.
        il.warmup_inner_layer(self.inner_classifier, input_shape)

        self.inner_vars = il.get_inner_variables(self.inner_classifier)
        print("Found inner vars:", self.inner_vars)

        # Collect mutable inner variables from inner network.
        # If the outer network outputs the inner variable per-step
        # then we can not mutate it.
        self.trainable_inner_vars = il.get_trainable_inner_variables(self.inner_classifier)

        #self.outer_network = OuterConvNetwork(self.inner_vars, num_inner_loops=self.num_inner_loops)
        self.outer_network = OuterSeperatedConstantNetwork(self.inner_vars, num_inner_loops=self.num_inner_loops)
        #self.outer_network = OuterConvNetwork(self.inner_vars, num_inner_loops=self.num_inner_loops)
        print("Num inner trainable vars:", self.outer_network.output_size)

    def get_loss(self, train_inputs, train_labels, test_inputs, test_labels):
        assert train_inputs.shape[2:] == test_inputs.shape[2:]
        assert len(train_labels.shape) == 2 and len(test_labels.shape) == 2
        assert train_inputs.shape[:2] == train_labels.shape[:2] and test_inputs.shape[:2] == test_labels.shape[:2]

        self._build(train_inputs.shape[2:])

        def _image_summary(name, images):
            tf.summary.image(name, tf.cast(255 * images[:, 0], tf.uint8))

        def _avg_scalar_summary(name, values):
            tf.summary.scalar(name, tf.reduce_mean(values))

        def _cosine_similarity_summary(name, a, b):
            a_reshaped = tf.reshape(a, (-1,))
            b_reshaped = tf.reshape(b, (-1,))
            normalized_a = a_reshaped / tf.sqrt(tf.reduce_sum(tf.square(a_reshaped)))
            normalized_b = b_reshaped / tf.sqrt(tf.reduce_sum(tf.square(b_reshaped)))
            similarity = tf.reduce_sum(normalized_a * normalized_b)
            tf.summary.scalar(name, similarity)
            return similarity

        # Keep initial and step test losses and average them
        # in the end.
        test_losses = []
        adjusted_losses = []

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
        def _get_inner_var(inner_var, batch_index, step):
            if batch_index == 0:
                print("Getting", inner_var.name, "for step", step)

            if inner_var.per_step:
                #print("Getting", inner_var.name, "at step", step)
                return self.outer_network.get_inner_variable(inner_var, step)[batch_index]
            else:
                # This will only work if the mutable_inner_vars correspond to the wanted step
                # since we are not using it in any way.
                return mutable_inner_vars[inner_var][batch_index]

        # Assign getters to the function above for the inner variables
        for inner_var in self.inner_vars:
            inner_var.getter = _get_inner_var

        # Test image summary
        _image_summary("test_input", test_inputs)
        _image_summary("train_input", train_inputs)
        tf.summary.image("test_input_mean", tf.cast(255 * tf.reduce_mean(test_inputs, axis=1), tf.uint8))
        tf.summary.image("train_input_mean", tf.cast(255 * tf.reduce_mean(train_inputs, axis=1), tf.uint8))

        previous_gradients = {}

        for step in range(self.num_inner_loops):
            print("------ Starting step", step)

            # Calculate train loss for this step
            il.set_inner_train_state(self.inner_classifier, is_train=True)
            il.set_inner_step(self.inner_classifier, step)
            train_loss, train_acc = self.inner_classifier.get_loss(train_inputs, train_labels)
            _avg_scalar_summary("train_loss_%d" % step, train_loss)
            _avg_scalar_summary("train_acc_%d" % step, train_acc)

            # Calculate test loss for this step
            il.set_inner_train_state(self.inner_classifier, is_train=False)
            test_loss, test_acc = self.inner_classifier.get_loss(test_inputs, test_labels)
            test_losses.append(test_loss)
            _avg_scalar_summary("test_loss_%d" % step, test_loss)
            _avg_scalar_summary("test_acc_%d" % step, test_acc)

            if self.adjust_loss:
                adjusted_losses.append(tf.maximum(train_loss, test_loss))
                _avg_scalar_summary("loss_adjusted_step_%d" % step, adjusted_losses[-1])

            # Mutable inner variable gradient update using train loss for this step
            similarities = []

            mutable_inner_vars_keys, mutable_inner_vars_values = list(mutable_inner_vars.keys()), list(mutable_inner_vars.values())
            mutable_inner_vars_grads = tf.gradients(train_loss, mutable_inner_vars_values)
            for inner_var, weights, grads in zip(mutable_inner_vars_keys, mutable_inner_vars_values, mutable_inner_vars_grads):
                lr = self.outer_network.get_learning_rate(inner_var, step)
                assert not inner_var.per_step
                if grads is not None:
                    # Prevent second order derivatives for first order training
                    if self.first_order:
                        grads = tf.stop_gradient(grads)

                    # Momentum
                    #if inner_var in previous_gradients:
                    #    grads = 0.9 * previous_gradients[inner_var] + grads

                    previous_gradients[inner_var] = grads

                    mutable_inner_vars[inner_var] = weights - lr * grads

                    similarity = _cosine_similarity_summary("cossim_%s_step_%d" % (inner_var.name, step), weights[0], weights[1])
                    similarities.append(similarity)
                else:
                    raise Exception("Grads none for %s (tensor: %s) (unused inner variable?)" % (inner_var.name, weights))

            tf.summary.scalar("cossim_avg_step_%d" % step, tf.reduce_mean(similarities))
        # Calculate final test loss
        print("------- Final evaluation step")
        il.set_inner_step(self.inner_classifier, self.num_inner_loops)
        il.set_inner_train_state(self.inner_classifier, is_train=True)
        train_loss, train_acc = self.inner_classifier.get_loss(train_inputs, train_labels) # Run on train set to get batch statistics
        _avg_scalar_summary("train_loss_final", train_loss)
        _avg_scalar_summary("train_acc_final", train_acc)
        il.set_inner_train_state(self.inner_classifier, is_train=False)
        test_loss, test_acc = self.inner_classifier.get_loss(test_inputs, test_labels)
        test_losses.append(test_loss)
        _avg_scalar_summary("test_loss_final", test_loss)
        _avg_scalar_summary("test_acc_final", test_acc)

        if self.adjust_loss:
            adjusted_losses.append(tf.maximum(train_loss, test_loss))
            _avg_scalar_summary("loss_adjusted_final", adjusted_losses[-1])

        test_loss_total = tf.reduce_mean(test_losses)
        _avg_scalar_summary("test_loss_total", test_loss_total)

        if self.adjust_loss:
            adjusted_loss_total = tf.reduce_mean(adjusted_losses)
            _avg_scalar_summary("adjusted_loss_total", adjusted_loss_total)

        # Final cosine similarity
        similarities = []
        for inner_var, weights in mutable_inner_vars.items():
            similarity = _cosine_similarity_summary("cossim_%s_final" % inner_var.name, weights[0], weights[1])
            similarities.append(similarity)
        tf.summary.scalar("cossim_avg_final", tf.reduce_mean(similarities))

        return adjusted_loss_total if self.adjust_loss else test_loss_total
