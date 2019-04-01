import tensorflow as tf


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

​
def batch_norm(inputs, training, name):
    return tf.layers.batch_normalization(
        inputs=inputs,
        axis=3,
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True,
        training=training,
        fused=True,
        name=name)


def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, name):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'),
        use_bias=False,
        kernel_initializer=tf.variance_scaling_initializer(),
        name=name)


def _building_block_v1(inputs, filters, training, projection_shortcut, strides):
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut,
                              training=training,
                              name="bns")

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        name="conv1")
    inputs = batch_norm(inputs, training, name="bn1")
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=1,
        name="conv2")
    inputs = batch_norm(inputs, training, name="bn2")
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides):
    shortcut = inputs

    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training,
                              name="bns")

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=1,
        name="conv1")
    inputs = batch_norm(inputs, training, "bn1")
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        name="conv2")
    inputs = batch_norm(inputs, training, "bn2")
    inputs = tf.nn.relu(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs,
        filters=4 * filters,
        kernel_size=1,
        strides=1,
        name="conv3")
    inputs = batch_norm(inputs, training, "bn3")
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name):

    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            name="convs")

    with tf.variable_scope("block0"):
        inputs = block_fn(inputs, filters, training,
                          projection_shortcut, strides)

    for i in range(1, blocks):
        with tf.variable_scope("block%d" % i):
            inputs = block_fn(inputs, filters, training, None, 1)

    return tf.identity(inputs, name)


class Model(object):

    def __init__(self,
                 resnet_size,
                 num_filters=64,
                 kernel_size=7,
                 conv_stride=2,
                 first_pool_size=3,
                 first_pool_stride=2,
                 block_strides=[1, 2, 2, 2],
                 global_pool=True,
                 final_fc=True):

        if resnet_size == 18:
            self.block_fn = _building_block_v1
            self.block_sizes = [2, 2, 2, 2]
            self.bottleneck = False
        elif resnet_size == 50:
            self.block_fn = _bottleneck_block_v1
            self.block_sizes = [3, 4, 6, 3]
            self.bottleneck = True

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_strides = block_strides
        self.global_pool = global_pool
        self.final_fc = final_fc

    def __call__(self, inputs, training):
        with tf.variable_scope('resnet', reuse=tf.AUTO_REUSE):
            end_points = {}

            print(inputs.shape)
            inputs = conv2d_fixed_padding(
                inputs=inputs,
                filters=self.num_filters,
                kernel_size=self.kernel_size,
                strides=self.conv_stride,
                name="convi")
            inputs = tf.identity(inputs, 'initial_conv')

            inputs = batch_norm(inputs, training, "bni")
            inputs = tf.nn.relu(inputs)
            print(inputs.shape)
            if self.first_pool_size:
                inputs = tf.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='SAME')
                inputs = tf.identity(inputs, 'initial_max_pool')
            print(inputs.shape)
            for i, num_blocks in enumerate(self.block_sizes):
                with tf.variable_scope("group%d" % (i + 1)):
                    num_filters = self.num_filters * (2**i)
                    inputs = block_layer(
                        inputs=inputs,
                        filters=num_filters,
                        bottleneck=self.bottleneck,
                        block_fn=self.block_fn,
                        blocks=num_blocks,
                        strides=self.block_strides[i],
                        training=training,
                        name='block_layer{}'.format(i + 1))

                    end_points['block%d' % i] = inputs
                print(inputs.shape)

            if self.global_pool:
                inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)

            if self.final_fc:
                inputs = tf.layers.conv2d(
                    inputs, 1024, 1, padding='VALID', name='final_fc')
                inputs = tf.squeeze(inputs, [1, 2])

            end_points['pre_logits'] = inputs

        return inputs, end_points


def resnet18():
    model = Model(18)
    return model


def resnet50():
    model = Model(50)
    return model
