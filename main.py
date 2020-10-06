import numpy as np


class NeuralNet:
    def __init__(self, structure):
        """Create an untrained neural network.

        :param int[] structure: The number of neurons in each layer from left to right,
            where the left-most "layer" is actually the input features and the
            right-most "layer" is actually the network's output.
        """
        self.structure = structure
        self.input_layer_size = structure[0]
        self.output_layer_size = structure[-1]

        # Create the arrays of weights and biases (to be optimized during training).
        self.weights = []
        self.biases = []
        for idx in range(len(structure) - 1):
            left_layer_size = structure[idx]
            right_layer_size = structure[idx + 1]
            # To map a 5-neuron layer's outputs to a 3-neuron layer's inputs, the weight
            # matrix is (3 x 5) and there are 3 bias values.
            layer_weights = np.empty((right_layer_size, left_layer_size))
            self.weights.append(layer_weights)
            layer_biases = np.empty((right_layer_size, 1))
            self.biases.append(layer_biases)

    def init_params(self):
        """Initialize weights and biases to reasonable starting values before training."""
        raise NotImplemented

    def train(self):
        """TODO: doc this"""
        raise NotImplemented

    def predict(self, inputs):
        """Use the network to predict the output of any number of inputs.

        :param np.ndarray inputs: m x n matrix. Each of n samples has m features.

        :returns np.ndarray outputs: q x n matrix. Each column is the output for the
            corresponding column of the input.
            q == self.structure[-1] (the size of the output layer, often 1)
        """
        # Make sure the input is a 2D ndarray (so we can handle an individual vector, or
        # a matrix of column vectors).
        if inputs.ndim < 2:
            inputs = inputs[:, np.newaxis]
        if inputs.ndim > 2:
            raise ValueError("Too many dimensions in inputs matrix.")
        # Make sure the inputs have the right number of features.
        if inputs.shape[0] != self.input_layer_size:
            msg = f"Samples must be length {self.input_layer_size} for this network."
            raise ValueError(msg)

        # Run the inputs through the network layer by layer.
        # Each column of inputs is 1 sample. Each column of layer_values corresponds to
        # 1 sample. Each of the following operations can be thought of as being applied
        # to each column independently.
        layer_values = inputs
        for idx in range(len(self.structure) - 1):
            # Apply the weights of the current layer to each column.
            layer_values = self.weights[idx].dot(layer_values)
            # Apply the biases of the current layer to each column.
            layer_values += self.biases[idx]
            # Apply the non-linear ReLU function to all elements. (Don't do this for the
            # final layer.)
            if idx < len(self.structure) - 1:
                layer_values *= layer_values > 0

        return layer_values


def main():
    x = NeuralNet([3, 5, 4, 1])

    inps = np.array([[1, 2], [3, 4], [5, 6]])

    y = x.predict(inps)

    print("Done.")


if __name__ == "__main__":
    main()
