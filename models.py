import tensorflow as tf
from resnet import softmax_layer, conv_layer, residual_block

n_dict = {20:1, 32:2, 44:3, 56:4}
# ResNet architectures used for CIFAR-10
def resnet(inpt, n, reuse=False):
    if n < 20 or (n - 20) % 12 != 0:
        print("ResNet depth invalid.")
        return

    num_conv = (n - 20) / 12 + 1
    layers = []

    with tf.variable_scope('root', reuse=reuse):

        with tf.variable_scope('conv1'):
            conv1 = conv_layer(inpt, [3, 3, 3, 8], 1)
            layers.append(conv1)

        for i in range (int(num_conv)):
            with tf.variable_scope('conv2_%d' % (i+1)):
                conv2_x = residual_block(layers[-1], 8, False)
                conv2 = residual_block(conv2_x, 8, False)
                layers.append(conv2_x)
                layers.append(conv2)

            assert conv2.get_shape().as_list()[1:] == [128, 128, 8]

        for i in range (int(num_conv)):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv3_%d' % (i+1)):
                conv3_x = residual_block(layers[-1], 16, down_sample)
                conv3 = residual_block(conv3_x, 16, False)
                layers.append(conv3_x)
                layers.append(conv3)

            assert conv3.get_shape().as_list()[1:] == [64, 64, 16]

        for i in range (int(num_conv)):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv4_%d' % (i+1)):
                conv4_x = residual_block(layers[-1], 32, down_sample)
                conv4 = residual_block(conv4_x, 32, False)
                layers.append(conv4_x)
                layers.append(conv4)

            assert conv4.get_shape().as_list()[1:] == [32, 32, 32]

        for i in range (int(num_conv)):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv5_%d' % (i+1)):
                conv5_x = residual_block(layers[-1], 64, down_sample)
                conv5 = residual_block(conv5_x, 64, False)
                layers.append(conv5_x)
                layers.append(conv5)

            assert conv5.get_shape().as_list()[1:] == [16, 16, 64]

        for i in range (int(num_conv)):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv6_%d' % (i+1)):
                conv6_x = residual_block(layers[-1], 128, down_sample)
                conv6 = residual_block(conv6_x, 128, False)
                layers.append(conv6_x)
                layers.append(conv6)

            assert conv6.get_shape().as_list()[1:] == [8, 8, 128]

        for i in range (int(num_conv)):
            down_sample = True if i == 0 else False
            with tf.variable_scope('conv7_%d' % (i+1)):
                conv7_x = residual_block(layers[-1], 256, down_sample)
                conv7 = residual_block(conv7_x, 256, False)
                layers.append(conv7_x)
                layers.append(conv7)

            assert conv7.get_shape().as_list()[1:] == [4, 4, 256]

        with tf.variable_scope('fc'):
            global_pool = tf.reduce_mean(layers[-1], [1, 2])
            assert global_pool.get_shape().as_list()[1:] == [256]

            out = softmax_layer(global_pool, [256, 108])
            layers.append(out)

    return layers[-1]
