from __future__ import absolute_import, division

import functools
import collections

import absl.flags as flag
import tensorflow as tf

import backbones.alexnet
import backbones.resnet
import backbones.resnet_ori
import utils
from backbones.alexnet import fcLayer

FLAGS = flag.FLAGS

def get_backbone():
    if FLAGS.backbone == 'alexnet':
        net = backbones.alexnet.alexnet

        if FLAGS.task in ['puzzle_train', 'puzzle_eval']:
            net = functools.partial(net, root_conv_stride=2)

        return net

    elif FLAGS.backbone == 'resnet50_v1':
        net = backbones.resnet_ori.resnet50()
#        net = functools.partial(net, mode="v1")

        return net

    elif FLAGS.backbone == 'resnet18_v1':
        net = backbones.resnet_ori.resnet18()
        return net

    else:
        pass

    return


class JPSModel(object):
    def __init__(self, data, is_training):
        self.hungarian_module = tf.load_op_library(
            '/tmp/work/munkres/hungarian.so')
        self.tower_size = FLAGS.batch_size // FLAGS.gpu_num

        self.indices_bottom = tf.reshape(tf.tile(tf.expand_dims(
            list(range(self.tower_size)), 1), [1, 9]), [-1]) * \
            (FLAGS.config**2)

        images = tf.reshape(
            data["image"], [-1, FLAGS.patch_size, FLAGS.patch_size, 3])
        labels = tf.reshape(data["label"], [-1])

        mean = tf.reduce_mean(images, [-2, -3], keepdims=True)
        images = images - mean

        backbone = get_backbone()
        features, _ = backbone(images, is_training)

        if FLAGS.binary:
            features_b = tf.reshape(
                features, [self.tower_size, 9, features.get_shape().as_list()[-1]])
            print(features.shape)
            print(features_b.shape)
            binary_loss_list = []

            for pair_i in range(FLAGS.config ** 2):
                for pair_j in range(FLAGS.config ** 2):

                    if pair_i == pair_j:
                        continue

                    pair_input = tf.concat(
                        [features_b[:, pair_i, :], features_b[:, pair_j, :]], axis=1)
                    pair_fc2 = fcLayer(
                        pair_input, 512, bias_init=1.0, name="fc2_binary")
                    pair_fc3 = fcLayer(
                        pair_fc2, 9, std_init=0.01, name="fc3_binary", reluFlag=False)

                    binary_one_hot = tf.one_hot(
                        [utils.pair_label_3[pair_i][pair_j]] * self.tower_size, 9)
                    binary_loss = tf.losses.softmax_cross_entropy(
                        binary_one_hot, pair_fc3)
                    binary_loss_list.append(binary_loss)
            mean_binary_loss = tf.reduce_mean(binary_loss_list)

        entropy_loss_list = []
        column_loss_list = []
        self.perm_list = [labels]

        # feature are firstly permuated by the input labels
        perm = labels
        for _ in range(FLAGS.iter_num):
            # reorder the features by indices
            indices = perm + self.indices_bottom
            gathered_features = tf.gather(features, indices)
            gathered_features = tf.reshape(
                gathered_features, [self.tower_size, -1])

            fc2 = fcLayer(gathered_features, 4096, bias_init=1.0, name="fc2")
            fc3 = fcLayer(fc2, FLAGS.config**4, std_init=0.01,
                          name="fc3", reluFlag=False)
            logits = tf.reshape(fc3, [self.tower_size,
                                      FLAGS.config**2,
                                      FLAGS.config**2])

            # loss function
            reshaped_perm = tf.reshape(
                perm, [self.tower_size, FLAGS.config**2])
            one_hot_perm = tf.one_hot(reshaped_perm, FLAGS.config**2)
            entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=one_hot_perm)

            prob = tf.nn.softmax(logits, axis=2)
            column_loss = tf.square(tf.reduce_sum(prob, axis=1) - 1)

            # predict permuation of the next iteration by hungarian algorithm
            predicted_perm = self.hungarian_module.hungarian(-1*prob)
            predicted_perm = tf.reshape(predicted_perm, [-1])

            # reorder the permutation for the next iteration based on the
            # current iteration
            predicted_indices = predicted_perm + self.indices_bottom
            predicted_indices = tf.reshape(predicted_indices, [-1, 1])
            new_perm = tf.scatter_nd(indices=predicted_indices,
                                     updates=perm,
                                     shape=[self.tower_size*(FLAGS.config**2)])
            perm = new_perm

            self.perm_list.append(perm)
            entropy_loss_list.append(entropy_loss)
            column_loss_list.append(column_loss)

        mean_column_loss = tf.reduce_mean(column_loss_list)
        mean_entropy_loss = tf.reduce_mean(entropy_loss_list)

        all_var = tf.trainable_variables()
        backbone_var = [var for var in all_var if (
            "alexnet" in var.name) or ("resnet" in var.name)]
        binary_var = [var for var in all_var if "binary" in var.name]
        unary_var = [var for var in all_var if (
            var not in backbone_var) and (var not in binary_var)]

        self.compute_gradients_losses = [{'value': mean_entropy_loss,
                                          'var_list': backbone_var + unary_var}]
        self.display_losses = [{'name': tf.convert_to_tensor('c_loss'), 'value': mean_column_loss},
                               {'name': tf.convert_to_tensor('e_loss'), 'value': mean_entropy_loss}]
        if FLAGS.binary:
            self.compute_gradients_losses.append(
                {'value': mean_binary_loss, 'var_list': backbone_var + binary_var})
            self.display_losses.append(
                {'name': tf.convert_to_tensor('b_loss'), 'value': mean_binary_loss})
