import tensorflow as tf
import numpy as np
import innerlayers as il

class OuterNetwork:
    def __init__(self, inner_variables, num_inner_loops):
        self.output = None
        self.inner_var_index = {}
        index = 0
        for inner_var in inner_variables:
            inner_var_size = np.prod(inner_var.shape)
            self.inner_var_index[inner_var] = []
            for step in range(num_inner_loops):
                self.inner_var_index[inner_var].append(index)
                if inner_var.per_step or step + 1 == num_inner_loops:
                    index += inner_var_size
        self.output_size = index
        self.num_inner_loops = num_inner_loops

    def get_inner_variable(self, inner_variable, step):
        """
        Gets the values for the inner variable at the specified step.
        Returns one variable for every outer batch.
        [OuterBatchSize, *VariableShape]
        """
        assert self.output is not None

        outer_batch_size = tf.shape(self.output)[0]
        start_index = self.inner_var_index[inner_variable][step]
        end_index = start_index + np.prod(inner_variable.shape)
        return tf.reshape(self.output[:, start_index:end_index], (outer_batch_size, *inner_variable.shape))

    def calculate_output(self, inputs):
        pass
