import tensorflow as tf

# An implementation of a Ladder Network
# Paper link: http://arxiv.org/pdf/1507.02672v2.pdf

def batch_norm(x):
    mean, variance = tf.nn.moments(x, axes=[0])

    x_hat = (x - mean)/ tf.sqrt(variance)

    return x_hat





class LadderNetworkMLP():

    def __init__(self):
        self.model = self.build_model()
        self.layers = []

    def build_model(self, layers_sizes, input_size, output_size, activation):
        X = tf.placeholder(tf.float32, shape=[None, input_size])
        y = tf.placeholder(tf.float32, shape=[None, output_size])

        # Parameters to learn for batch normalization
        # Beta is the scaling parameter
        # Gamma is the bias
        # These parameters are redundant for linear activation functions
        # such as ReLU.
        # From page 6

        beta = tf.Variable()
        gamma = tf.Variable()


        # Create the layers for the network



        # Create the first layer

        W = tf.Variable(tf.zeros([input_size, layers_sizes[0]]))
        b = tf.Variable(tf.zeros(layers_sizes[0]))

        # z_tilda is the preactivation of each layer
        z_tilda = tf.Variable(tf.zeros()) + tf.random_normal()

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

            # Batch normalization and add isotropic gaussian noise
            # From page 5, Algorithm 1

            batch_normalized_output = batch_norm(layer_mat_mul)
            z_tilda = batch_normalized_output + tf.random_normal(batch_normalized_output.getshape())

            # Activation at each layer
            h = activation(tf.mul(gamma, (z_tilda + beta)))


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


        # Clean encoder (for denoising targets)
        # See Algorithm 1
        h_0 = X


        for i, layer in enumerate(self.layers):
            layer_weights
            W = layer['W']
            b = layer['b']

            if i == 0:
                h = h_0
            else:
                h = self.layers[i-1]

            # Calculate the preactivation for the layer z

            z = tf.matmul(W, h)

            # mu
            # sigma
            # z_norm
            # h activation



        for layer_i, layer in enumerate(self.layers):
            if layer_i == 1:
                u = batch_norm(h)
            else:
                u = batch_norm()



        # Decoder and denoising

        for

        return output




