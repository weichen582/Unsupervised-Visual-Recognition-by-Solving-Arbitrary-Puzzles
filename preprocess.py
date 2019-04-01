import functools
import absl.flags as flags

import tensorflow as tf

import utils

FLAGS = flags.FLAGS


def __crop(image, is_training, crop_size):
    h, w, c = crop_size[0], crop_size[1], image.shape[-1]
    if is_training:
        return tf.random_crop(image, [h, w, c])
    else:
        dy = (tf.shape(image)[0] - h) // 2
        dx = (tf.shape(image)[1] - w) // 2
        return tf.image.crop_to_bounding_box(image, dy, dx, h, w)


def get_resize_small(smaller_size):
    """Resizes the smaller side to `smaller_size` keeping aspect ratio."""
    def _resize_small(data):
        image = data["image"]
        h, w = tf.shape(image)[-3], tf.shape(image)[-2]

        # Figure out the necessary h/w.
        ratio = tf.to_float(smaller_size) / tf.to_float(tf.minimum(h, w))
        h = tf.to_int32(tf.round(tf.to_float(h) * ratio))
        w = tf.to_int32(tf.round(tf.to_float(w) * ratio))

        static_rank = len(image.get_shape().as_list())
        if static_rank == 3:  # A single image: HWC
            data["image"] = tf.image.resize_images(image[None], [h, w])[0]
        elif static_rank == 4:  # A batch of images: BHWC
            data["image"] = tf.image.resize_images(image, [h, w])
        return data
    return _resize_small


def get_resize(im_size, randomize_resize_method=False):
    def _resize(image, method, align_corners):
        def _process():
            # The resized_images are of type float32 and might fall outside of range
            # [0, 255].
            resized = tf.cast(
                tf.image.resize_images(
                    image, im_size, method, align_corners=align_corners),
                dtype=tf.float32)
            return resized
        return _process

    def _resize_pp(data):
        im = data["image"]
        if randomize_resize_method:
            # pick random resizing method
            r = tf.random_uniform([], 0, 3, dtype=tf.int32)
            im = tf.case({
                tf.equal(r, tf.cast(0, r.dtype)):
                    _resize(im, tf.image.ResizeMethod.BILINEAR, True),
                tf.equal(r, tf.cast(1, r.dtype)):
                    _resize(im, tf.image.ResizeMethod.NEAREST_NEIGHBOR, True),
                tf.equal(r, tf.cast(2, r.dtype)):
                    _resize(im, tf.image.ResizeMethod.BICUBIC, True),
                tf.equal(r, tf.cast(3, r.dtype)):
                    _resize(im, tf.image.ResizeMethod.AREA, False),
            })
        else:
            im = tf.image.resize_images(im, im_size)
        data["image"] = im
        return data
    return _resize_pp


def get_crop(is_training, crop_size):
    """Returns a random (or central at test-time) crop of `crop_size`."""
    def _crop_pp(data):
        crop_fn = functools.partial(
            __crop, is_training=is_training, crop_size=crop_size)
        data["image"] = utils.tf_apply_to_image_or_images(
            crop_fn, data["image"])
        return data
    return _crop_pp


def get_random_flip_lr(is_training):
    def _random_flip_lr_pp(data):
        if is_training:
            data["image"] = utils.tf_apply_to_image_or_images(
                tf.image.random_flip_left_right, data["image"])
        return data
    return _random_flip_lr_pp


def get_random_rotation(is_training):
    def _rotation_pp(data):
        if is_training:
            data['image'] = utils.tf_apply_to_image_or_images(
                lambda img: tf.image.rot90(
                    img, k=tf.random_uniform([], dtype=tf.int32, maxval=4)),
                data['image'])
        return data
    return _rotation_pp


def get_standardization_preprocess():
    def _standardization_pp(data):
        # Trick: normalize each patch to avoid low level statistics.
        data["image"] = utils.tf_apply_to_image_or_images(
            tf.image.per_image_standardization, data["image"])
        return data
    return _standardization_pp


def get_to_gray_preprocess(grayscale_probability):
    def _to_gray(image):
        # Transform to grayscale by taking the mean of RGB.
        return tf.tile(tf.reduce_mean(image, axis=2, keepdims=True), [1, 1, 3])

    def _to_gray_pp(data):
        data["image"] = utils.tf_apply_to_image_or_images(
            lambda img: utils.tf_apply_with_probability(
                grayscale_probability, _to_gray, img),
            data["image"])
        return data
    return _to_gray_pp


def get_crop_patches_fn(is_training, split_per_side, channel_jitter=0):
    """Gets a function which crops split_per_side x split_per_side patches.
    Args:
        is_training: is training flag.
        split_per_side: split of patches per image side.
        patch_jitter: jitter of each patch from each grid. E.g. 255x255 input
        image with split_per_side=3 will be split into 3 85x85 grids, and
        patches are cropped from each grid with size (grid_size-patch_jitter,
        grid_size-patch_jitter).
    Returns:
        A function returns name to tensor dict. This function crops split_per_side x
        split_per_side patches from "image" tensor in input data dict.
    """

    def _image_to_patches(image, is_training, split_per_side, channel_jitter=0):
        """Crops split_per_side x split_per_side patches from input image.
        Args:
            image: input image tensor with shape [h, w, c].
            is_training: is training flag.
            split_per_side: split of patches per image side.
            patch_jitter: jitter of each patch from each grid.
        Returns:
            Patches tensor with shape [patch_count, hc, wc, c].
        """
        patches = []
        for i in range(split_per_side):
            for j in range(split_per_side):
                p = tf.image.crop_to_bounding_box(image,
                                                  i * FLAGS.cell_size,
                                                  j * FLAGS.cell_size,
                                                  FLAGS.cell_size,
                                                  FLAGS.cell_size)

                # Trick: crop a small tile from pixel cell, to avoid edge continuity.
                if channel_jitter > 0:
                    channels = []
                    p = __crop(p, is_training,
                               [FLAGS.patch_size + channel_jitter,
                                FLAGS.patch_size + channel_jitter])
                    for k in range(3):
                        c = __crop(p[..., k:k+1], is_training,
                                   [FLAGS.patch_size, FLAGS.patch_size])
                        channels.append(c)
                    p = tf.concat(channels, -1)
                else:
                    p = __crop(p, is_training, [
                               FLAGS.patch_size, FLAGS.patch_size])
                patches.append(p)
        return tf.stack(patches)

    def _crop_patches_pp(data):
        image = data["image"]

        image_to_patches_fn = functools.partial(
            _image_to_patches,
            is_training=is_training,
            split_per_side=split_per_side,
            channel_jitter=channel_jitter)
        image = utils.tf_apply_to_image_or_images(image_to_patches_fn, image)
        data["image"] = image
        return data
    return _crop_patches_pp
