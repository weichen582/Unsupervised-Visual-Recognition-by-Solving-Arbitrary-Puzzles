import collections
import os

import numpy as np
import tensorflow as tf

pair_label_3 = [[-1, 5, 0, 7, 8, 0, 0, 0, 0],
                [4, -1, 5, 6, 7, 8, 0, 0, 0],
                [0, 4, -1, 0, 6, 7, 0, 0, 0],
                [2, 3, 0, -1, 5, 0, 7, 8, 0],
                [1, 2, 3, 4, -1, 5, 6, 7, 8],
                [0, 1, 2, 0, 4, -1, 0, 6, 7],
                [0, 0, 0, 2, 3, 0, -1, 5, 0],
                [0, 0, 0, 1, 2, 3, 4, -1, 5],
                [0, 0, 0, 0, 1, 2, 0, 4, -1]]


def hamming_dist(a, b):
    count = np.shape(np.nonzero(a - b))[1]
    return float(count)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def save(sess, saver, iter_num, ckpt_dir, model_name='JPS'):
    checkpoint_dir = ckpt_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("[*] Saving model...")
    saver.save(sess,
               os.path.join(checkpoint_dir, model_name),
               global_step=iter_num)


def load(sess, saver, checkpoint_dir):
    print("[*] Reading checkpoint...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:  # pylint: disable=no-member
        full_path = tf.train.latest_checkpoint(checkpoint_dir)
        global_step = int(full_path.split('/')[-1].split('-')[-1])
        saver.restore(sess, full_path)
        return True, global_step
    else:
        return False, 0


def tf_apply_with_probability(p, fn, x):
    """Apply function `fn` to input `x` randomly `p` percent of the time."""
    return tf.cond(
        tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), p),
        lambda: fn(x),
        lambda: x)


def tf_apply_to_image_or_images(fn, image_or_images):
    """Applies a function to a single image or each image in a batch of them.
    Args:
      fn: the function to apply, receives an image, returns an image.
      image_or_images: Either a single image, or a batch of images.
    Returns:
      The result of applying the function to the image or batch of images.
    Raises:
      ValueError: if the input is not of rank 3 or 4.
    """
    static_rank = len(image_or_images.get_shape().as_list())
    if static_rank == 3:  # A single image: HWC
        return fn(image_or_images)
    elif static_rank == 4:  # A batch of images: BHWC
        return tf.map_fn(fn, image_or_images)
    elif static_rank > 4:  # A batch of images: ...HWC
        input_shape = tf.shape(image_or_images)
        h, w, c = image_or_images.get_shape().as_list()[-3:]
        image_or_images = tf.reshape(image_or_images, [-1, h, w, c])
        image_or_images = tf.map_fn(fn, image_or_images)
        return tf.reshape(image_or_images, input_shape)
    else:
        raise ValueError("Unsupported image rank: %d" % static_rank)


def str2intlist(s, repeats_if_single=None):
    """Parse a config's "1,2,3"-style string into a list of ints.
    Args:
      s: The string to be parsed, or possibly already an int.
      repeats_if_single: If s is already an int or is a single element list,
                         repeat it this many times to create the list.
    Returns:
      A list of integers based on `s`.
    """
    if isinstance(s, int):
        result = [s]
    else:
        result = [int(i.strip()) if i != "None" else None
                  for i in s.split(",")]
    if repeats_if_single is not None and len(result) == 1:
        result *= repeats_if_single
    return result


def adaptive_pool(inp, num_target_dimensions=9000, mode="adaptive_max"):
    """Adaptive pooling layer.
       This layer performs adaptive pooling, such that the total
       dimensionality of output is not bigger than num_target_dimension
    Args:
       inp: input tensor
       num_target_dimensions: maximum number of output dimensions
       mode: one of {"adaptive_max", "adaptive_avg", "max", "avg"}
    Returns:
      Result of the pooling operation
    Raises:
      ValueError: mode is unexpected.
    """

    size, _, k = inp.get_shape().as_list()[1:]
    if mode in ["adaptive_max", "adaptive_avg"]:
        if mode == "adaptive_max":
            pool_fn = tf.nn.fractional_max_pool
        else:
            pool_fn = tf.nn.fractional_avg_pool

        # Find the optimal target output tensor size
        target_size = (num_target_dimensions / float(k)) ** 0.5
        if (abs(num_target_dimensions - k * np.floor(target_size) ** 2) <
                abs(num_target_dimensions - k * np.ceil(target_size) ** 2)):
            target_size = max(np.floor(target_size), 1.0)
        else:
            target_size = max(np.ceil(target_size), 1.0)

        # Get optimal stride. Subtract epsilon to ensure correct rounding in
        # pool_fn.
        stride = size / target_size - 1.0e-5

        # Make sure that the stride is valid
        stride = max(stride, 1)
        stride = min(stride, size)

        result = pool_fn(inp, [1, stride, stride, 1])[0]
    elif mode in ["max", "avg"]:
        if mode == "max":
            pool_fn = tf.contrib.layers.max_pool2d
        else:
            pool_fn = tf.contrib.layers.avg_pool2d
        total_size = float(np.prod(inp.get_shape()[1:].as_list()))
        stride = int(np.ceil(np.sqrt(total_size / num_target_dimensions)))
        stride = min(max(1, stride), size)

        result = pool_fn(inp, kernel_size=stride, stride=stride)
    else:
        raise ValueError("Not supported %s pool." % mode)

    return result


def apply_fractional_pooling(taps, target_features=9000, mode='adaptive_max'):
    """Applies fractional pooling to each of `taps`.
    Args:
      taps: A dict of names:tensors to which to attach the head.
      target_features: If the input tensor has more than this number of features,
                       perform fractional pooling to reduce it to this amount.
      mode: one of {'adaptive_max', 'adaptive_avg', 'max', 'avg'}
    Returns:
      tensors: An ordered dict with target_features dimension tensors.
    Raises:
      ValueError: mode is unexpected.
    """
    out_tensors = collections.OrderedDict()
    for k, t in sorted(taps.items()):
        if len(t.get_shape().as_list()) == 2:
            t = t[:, None, None, :]
        _, h, w, num_channels = t.get_shape().as_list()
        if h * w * num_channels > target_features:
            t = adaptive_pool(t, target_features, mode)
        out_tensors[k] = t

    return out_tensors
