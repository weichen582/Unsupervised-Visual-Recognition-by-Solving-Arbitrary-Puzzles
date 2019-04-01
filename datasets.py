from __future__ import absolute_import, division, print_function

import functools
import glob
import os

import absl.flags as flags
import tensorflow as tf

from preprocess import *

FLAGS = flags.FLAGS

def _decode_image(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)

    data = {}

    data['image'] = image
    data['label'] = label

    return data


def _preprocess_patch(data, is_training):
    def expand(fn_name):
        if fn_name == "plain_preprocess":
            yield lambda x: x
        elif fn_name == "resize":
            yield get_resize(
                utils.str2intlist(FLAGS.resize_size, 2),
                False)
        elif fn_name == "resize_small":
            yield get_resize_small(FLAGS.smaller_size)
        elif fn_name == "crop":
            yield get_crop(is_training,
                           utils.str2intlist(FLAGS.crop_size, 2))
        elif fn_name == "central_crop":
            yield get_crop(False, utils.str2intlist(FLAGS.crop_size, 2))
        elif fn_name == "flip_lr":
            yield get_random_flip_lr(is_training)
        elif fn_name == "random_rotation":
            yield get_random_rotation(is_training)
        elif fn_name == "gray":
            yield get_to_gray_preprocess(FLAGS.grayscale_prob)
        elif fn_name == "crop_patches":
            yield get_crop_patches_fn(
                is_training,
                split_per_side=FLAGS.config,
                channel_jitter=FLAGS.get_flag_value("channel_jitter", 0))
        elif fn_name == "standardization":
            yield get_standardization_preprocess()
        else:
            raise ValueError("Not supported preprocessing %s" % fn_name)

    # Apply all the individual steps in sequence.
    for fn_name in FLAGS.preprocessing.split(","):
        for p in expand(fn_name.strip()):
            data = p(data)
            print("Data after %s" % p)

    if FLAGS.task in ['puzzle_train', 'puzzle_eval']:
        data['label'] = tf.random_shuffle(list(range(FLAGS.config**2)))

    return data


def path_process_txt(paths_file):
    paths = open(paths_file).readlines()
    filenames = [line.strip().split(' ')[0] for line in paths]
    labels = [int(line.strip().split(' ')[1]) for line in paths]
    return filenames, labels


def data_loader(is_training):
    if is_training:
        paths_file = FLAGS.train_paths_file
    else:
        paths_file = FLAGS.val_paths_file

    file_format = os.path.splitext(paths_file)[-1]
    if file_format == '.txt':
        filenames, labels = path_process_txt(paths_file)
    elif file_format == '.csv':
        pass
    else:
        pass

    num_batch = len(filenames) // FLAGS.batch_size

    _preprocess_patch_fn = functools.partial(
        _preprocess_patch, is_training=is_training)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    if is_training:
        dataset = dataset.shuffle(len(filenames))

    dataset = dataset.map(_decode_image, num_parallel_calls=10)
    dataset = dataset.map(_preprocess_patch_fn, num_parallel_calls=30)
    dataset = dataset.repeat().batch(FLAGS.batch_size)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_initializable_iterator()
    loader_init_op = iterator.initializer

    return iterator, loader_init_op, num_batch
