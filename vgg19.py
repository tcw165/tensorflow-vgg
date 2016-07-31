import time

import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    TRAIN_MODE = 0
    TEST_MODE = 1

    WIDTH = 224
    HEIGHT = 224
    CHANNELS = 3

    _model = None

    _inputRGB = None
    _inputBGR = None
    _inputNormalizedBGR = None

    _conv1_1 = None
    _conv1_2 = None
    _pool1 = None

    _conv2_1 = None
    _conv2_2 = None
    _pool2 = None

    _conv3_1 = None
    _conv3_2 = None
    _conv3_3 = None
    _conv3_4 = None
    _pool3 = None

    _conv4_1 = None
    _conv4_2 = None
    _conv4_3 = None
    _conv4_4 = None
    _pool4 = None

    _conv5_1 = None
    _conv5_2 = None
    _conv5_3 = None
    _conv5_4 = None
    _pool5 = None

    _fc6 = None
    _relu6 = None

    _fc7 = None
    _relu7 = None

    _fc8 = None

    _preds = None

    _loss = None
    _optimizer = None

    def __init__(self, model=None):
        """
        :param model: The model either for back-propagation or
        forward-propagation.
        """
        self._model = model

        # Define the input placeholder with RGB channels.
        self._inputRGB = tf.placeholder(tf.float32,
                                        [None,
                                         Vgg19.WIDTH,
                                         Vgg19.HEIGHT,
                                         Vgg19.CHANNELS])

        # Convert RGB to BGR order
        red, green, blue = tf.split(3, 3, self._inputRGB)
        self._inputBGR = tf.concat(3, [
            blue,
            green,
            red,
        ])

        # normalize the input so that the elements all have nearly equal
        # variances.
        self._inputNormalizedBGR = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        # Setup the VGG-Net graph.
        self._conv1_1 = self.conv_layer(self._inputNormalizedBGR, "conv1_1")
        self._conv1_2 = self.conv_layer(self._conv1_1, "conv1_2")
        self._pool1 = self.max_pool(self._conv1_2, 'pool1')

        self._conv2_1 = self.conv_layer(self._pool1, "conv2_1")
        self._conv2_2 = self.conv_layer(self._conv2_1, "conv2_2")
        self._pool2 = self.max_pool(self._conv2_2, 'pool2')

        self._conv3_1 = self.conv_layer(self._pool2, "conv3_1")
        self._conv3_2 = self.conv_layer(self._conv3_1, "conv3_2")
        self._conv3_3 = self.conv_layer(self._conv3_2, "conv3_3")
        self._conv3_4 = self.conv_layer(self._conv3_3, "conv3_4")
        self._pool3 = self.max_pool(self._conv3_4, 'pool3')

        self._conv4_1 = self.conv_layer(self._pool3, "conv4_1")
        self._conv4_2 = self.conv_layer(self._conv4_1, "conv4_2")
        self._conv4_3 = self.conv_layer(self._conv4_2, "conv4_3")
        self._conv4_4 = self.conv_layer(self._conv4_3, "conv4_4")
        self._pool4 = self.max_pool(self._conv4_4, 'pool4')

        self._conv5_1 = self.conv_layer(self._pool4, "conv5_1")
        self._conv5_2 = self.conv_layer(self._conv5_1, "conv5_2")
        self._conv5_3 = self.conv_layer(self._conv5_2, "conv5_3")
        self._conv5_4 = self.conv_layer(self._conv5_3, "conv5_4")
        self._pool5 = self.max_pool(self._conv5_4, 'pool5')

        self._fc6 = self.fc_layer(self._pool5, "fc6")
        self._relu6 = tf.nn.relu(self._fc6)

        self._fc7 = self.fc_layer(self._relu6, "fc7")
        self._relu7 = tf.nn.relu(self._fc7)

        self._fc8 = self.fc_layer(self._relu7, "fc8")

        # For forward propagation.
        self._preds = tf.nn.softmax(self._fc8, name="prediction")

        # For backward propagation.
        # self._loss = tf.nn.softmax_cross_entropy_with_logits()

    @property
    def inputRGB(self):
        """
        :return: The input RGB images tensor of channels in RGB order.
        """
        return self._inputRGB

    @property
    def inputBGR(self):
        """
        :return: The input RGB images tensor of channels in BGR order.
        """
        return self._inputBGR

    @property
    def preds(self):
        """
        :return: The prediction(s) tensor.
        """
        return self._preds

    @property
    def optimizer(self):
        """
        :return: The optimizer tensor.
        """
        return self._optimizer

    def avg_pool(self, value, name):
        return tf.nn.avg_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def max_pool(self, value, name):
        return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def conv_layer(self, value, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(value, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, value, name):
        with tf.variable_scope(name):
            shape = value.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(value, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self._model[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self._model[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self._model[name][0], name="weights")
