import tensorflow as tf

# An implementation of a Ladder Network
# Paper link: http://arxiv.org/pdf/1507.02672v2.pdf

def batch_norm(x, h):
    mean, variance = tf.nn.moments(x, axes=[0])






class LadderNetworkMLP():

    def __init__(self):
        self.model = self.build_model()
        self.layers = []

    def build_model(self, layers_sizes, input_size, output_size, activation):
        X = tf.placeholder(tf.float32, shape=[None, input_size])
        y = tf.placeholder(tf.float32, shape=[None, output_size])

        # Parameters to learn for batch normalization
        beta = tf.Variable()
        gamma = tf.Variable()


        # Create the layers for the network

        # Create the first layer

        W = tf.Variable(tf.zeros([input_size, layers_sizes[0]]))
        b = tf.Variable(tf.zeros(layers_sizes[0]))

        first_layer_mat = tf.matmul(X, W) + b

        first_layer_activation = tf.nn.relu(first_layer_mat)
        self.layers.append(first_layer_activation)

        for layer_i, layer_size in enumerate(layers_sizes):
            if layer_i == 0:
                W = tf.Variable(tf.zeros([first_layer_activation, layer_size]))
                b = tf.Variable([layer_size])

            elif layer_i == (len(layers_sizes) - 1):

                W = tf.Variable(tf.zeros([layer_size, output_size]))
                b = tf.Variable(tf.zeros([output_size]))

            else:
                W = tf.Variable(tf.zeros([layers_sizes[layer_i-1], layers_sizes[layer_i]]))
                b = tf.Variable(tf.zeros([layers_sizes[layer_i]]))

            # Calculate current layer
            layer_mat_mul = tf.matmul(self.layers[layer_i - 1], W) + b

            # Batch normalization

            mean, variance = tf.nn.moments(layer_mat_mul)



            layer_activation = tf.nn.relu(layer_mat_mul)

            # Save the weights and bias variables
            layer_weights = {}
            layer_weights['W'] = W
            layer_weights['b'] = b
            layer_weights['z'] = layer_activation

            self.layers.append(layer_weights)

        W = tf.Variable(tf.zeros([output_size]))
        b = tf.Variable(tf.zeros([output_size]))

        output_layer_weights = {}
        output_layer_weights['W'] = W
        output_layer_weights['b'] = b
        output_layer_weights['z'] = output_layer_weights

        output_mat_mul = tf.matmul(self.layers[-1], W) + b

        self.layers.append(output_layer_weights)

        output = tf.nn.softmax(output_mat_mul)

        return output




