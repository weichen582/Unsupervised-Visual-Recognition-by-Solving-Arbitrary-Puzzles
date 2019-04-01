import tensorflow as tf


def maxPoolLayer(x, ksize, stride, padding='VALID', name=None):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding,
                          name=name)


def LRN(x, R=2, alpha=2e-5, beta=0.75, bias=1.0, name=None):
    return tf.nn.local_response_normalization(x,
                                              depth_radius=R,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)


def fcLayer(x, outputD, name, std_init=0.005, bias_init=0.0, reluFlag=True):
    inputD = int(x.get_shape()[-1])
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable("w",
                            shape=[inputD, outputD],
                            dtype="float",
                            initializer=tf.random_normal_initializer(
                                stddev=std_init),
                            regularizer=tf.contrib.layers.l2_regularizer(5e-4))
        b = tf.get_variable("b",
                            [outputD],
                            dtype="float",
                            initializer=tf.constant_initializer(bias_init))
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)

        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out


def convLayer(x, ksize, stride, feature, padding='SAME', bias_init=0.0, groups=1, name=None):
    channel = int(x.get_shape()[-1])

    def conv(a, b): return tf.nn.conv2d(a,
                                        b,
                                        strides=[1, stride, stride, 1],
                                        padding=padding)

    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w",
                            shape=[ksize, ksize, channel / groups, feature],
                            initializer=tf.random_normal_initializer(
                                stddev=0.01),
                            regularizer=tf.contrib.layers.l2_regularizer(5e-4))
        b = tf.get_variable("b",
                            shape=[feature],
                            initializer=tf.constant_initializer(bias_init))

        xNew = tf.split(value=x, num_or_size_splits=groups, axis=3)
        wNew = tf.split(value=w, num_or_size_splits=groups, axis=3)

        featureMap = [conv(t1, t2) for t1, t2 in zip(xNew, wNew)]
        mergeFeatureMap = tf.concat(values=featureMap, axis=3)

        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(out, name=scope.name)


def alexnet(input, is_training, root_conv_stride=4):
    end_points = {}

    with tf.variable_scope('alexnet', reuse=tf.AUTO_REUSE):
        conv1 = convLayer(input, 11, root_conv_stride,
                          96, "VALID", name="conv1")
        end_points['conv1'] = conv1

        pool1 = maxPoolLayer(conv1, 3, 2, name="pool1")
        lrn1 = LRN(pool1, name="lrn1")

        conv2 = convLayer(lrn1, 5, 1, 256, groups=2,
                          bias_init=1.0, name="conv2")
        end_points['conv2'] = conv2

        pool2 = maxPoolLayer(conv2, 3, 2, name="pool2")
        lrn2 = LRN(pool2, name="lrn2")

        conv3 = convLayer(lrn2, 3, 1, 384, name="conv3")
        end_points['conv3'] = conv3

        conv4 = convLayer(conv3, 3, 1, 384, groups=2,
                          bias_init=1.0, name="conv4")
        end_points['conv4'] = conv4

        conv5 = convLayer(conv4, 3, 1, 256, groups=2,
                          bias_init=1.0, name="conv5")
        end_points['conv5'] = conv5

        conv5 = tf.pad(conv5, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]])
        pool5 = maxPoolLayer(conv5, 3, 2, name="pool5")

        #fc1 = fcLayer(tf.layers.flatten(pool5), 1024, name="fc6")
        fc1 = convLayer(pool5, 3, 1, 1024, "VALID", bias_init=1.0, name="fc6")
        end_points['fc6'] = fc1

    return fc1, end_points
