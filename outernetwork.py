import tensorflow as tf
import numpy as np
import innerlayers as il

class OuterNetwork:
    def __init__(self, inner_variables, num_inner_loops, fixed_lr=None):
        self.fixed_lr = fixed_lr
        self.output = None
        self.inner_var_index = {}
        self.inner_var_lr_index = {}
        index = 0
        num_vars = num_inner_loops + 1 # We need one more variable than steps since the final network will need variables too
        for inner_var in inner_variables:
            inner_var_size = np.prod(inner_var.shape)
            self.inner_var_index[inner_var] = []
            
            for step in range(num_vars):
                self.inner_var_index[inner_var].append(index)
                if inner_var.per_step or step + 1 == num_vars:
                    index += inner_var_size

            if fixed_lr is None and not inner_var.per_step:
                # Learning rate per step
                self.inner_var_lr_index[inner_var] = index
                index += num_inner_loops
        
        self.output_size = index
        self.num_inner_loops = num_inner_loops

    def get_inner_variable(self, inner_variable, step):
        """
        Gets the values for the inner variable at the specified step.
        Returns one variable for every outer batch.
        [OuterBatchSize, *VariableShape]
        """
        assert self.output is not None and step >= 0 and step <= self.num_inner_loops

        outer_batch_size = tf.shape(self.output)[0]
        start_index = self.inner_var_index[inner_variable][step]
        end_index = start_index + np.prod(inner_variable.shape)
        return tf.reshape(self.output[:, start_index:end_index], (outer_batch_size, *inner_variable.shape))

    def get_learning_rate(self, inner_variable, step):
        assert self.output is not None and not inner_variable.per_step and step >= 0 and step < self.num_inner_loops

        if self.fixed_lr is None:
            outer_batch_size = tf.shape(self.output)[0]
            index = self.inner_var_lr_index[inner_variable] + step
            return tf.reshape(self.output[:, index], (outer_batch_size, *([1] * len(inner_variable.shape))))
        else:
            return self.fixed_lr

    def calculate_output(self, inputs):
        pass
