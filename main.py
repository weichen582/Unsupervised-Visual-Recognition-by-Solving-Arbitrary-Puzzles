import datasets
import utils
import models
from logger import Logger
import tensorflow as tf
import numpy as np
import absl.flags as flags
import os
import time
import functools
from datetime import timedelta
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
​

FLAGS = flags.FLAGS

flags.DEFINE_string('task', None, 'Can be one of puzzle_train, puzzle_eval, '
                                  'linear_eval')
flags.mark_flag_as_required('task')
flags.DEFINE_string('experiment_name', None, 'Name for an experiment.')
flags.mark_flag_as_required('experiment_name')


flags.DEFINE_string('dataset', 'imagenet', 'Can be imagenet, places205')
flags.DEFINE_string('train_paths_file', 'files/train_cls.txt',
                    'The file which has all images and their labels')
flags.DEFINE_string('val_paths_file', 'files/val_sub_cls.txt',
                    'The file which has all images and their labels')
flags.DEFINE_string('logger_level', 'debug',
                    'To which level the logger keep the information.')


flags.DEFINE_integer('gpu_num', 1, 'Number of GPUs to use.')


flags.DEFINE_string('preprocessing', None,
                    'A comma-separated list of pre-processing steps to perform, '
                    'see preprocess.py.')
flags.mark_flag_as_required('preprocessing')


flags.DEFINE_integer('config', 3, 'How many times to split a side.')
flags.DEFINE_bool('binary', False, 'If use binary term.')
flags.DEFINE_integer('iter_num', 20, 'How many times to reorder the puzzle.')
flags.DEFINE_integer('resize_size', 224,
                     'For preprocess resize_size.')
flags.DEFINE_integer('smaller_size', 256,
                     'For preprocess resize_small.')
flags.DEFINE_integer('crop_size', 255,
                     'For preprocess crop and central_crop.')
flags.DEFINE_integer('patch_size', 64,
                     'For puzzle training. The size of each patch.')
flags.DEFINE_integer('cell_size', 85,
                     'For puzzle training. The size of each cell.')
flags.DEFINE_integer('channel_jitter', 5,
                     'For preprocess crop patches. Jitter for RGB channels.')
flags.DEFINE_float('grayscale_prob', 0.66,
                   'For preprocess gray. The probability to change on image to '
                   'gray scale.')


flags.DEFINE_string('backbone', 'alexnet', 'Can be alexnet, resnet50_v1')
flags.DEFINE_integer('batch_size', 256, 'Number of images in one batch.')
flags.DEFINE_float('base_lr', 0.01,
                   'The base learning-rate to use for training.')
flags.DEFINE_float('weight_decay', 1e-4, 'Strength of weight-decay.')
flags.DEFINE_integer('epochs', 90, 'Number of epochs to run training.')
flags.DEFINE_integer('save_every_epoch', 4,
                     'Save a checkpoint after X epochs.')
flags.DEFINE_integer('eval_every_epoch', 2,
                     'Evaluate the model after X epochs during training.')


def eval_puzzle(model_list, sess, num_batch, logger):
    ham_dist_count = np.zeros(21)
    acc_all_count = np.zeros(21)
    acc_2_count = np.zeros(21)

    logger.logger.debug('[*] All %d testing batches' % num_batch)

    for batch_id in range(num_batch):
        logger.logger.debug("batch %d =================" % batch_id)
        for model in model_list:
            perm_list_eval = sess.run(model.perm_list)

            for m in range(FLAGS.batch_size // FLAGS.gpu_num):
                for n in range(FLAGS.iter_num):
                    perm = perm_list_eval[n][m * FLAGS.config**2:
                                             (m + 1) * FLAGS.config**2]
                    ham_dist = utils.hamming_dist(perm,
                                                  np.arange(FLAGS.config**2))
                    ham_dist_count[n] += ham_dist

                    if ham_dist == 0:
                        acc_all_count[n] += 1.0
                    if ham_dist <= 2:
                        acc_2_count[n] += 1.0

        logger.logger.debug('[*] ave_hamming_dist %s' %
                            np.array2string(ham_dist_count / ((batch_id+1) * FLAGS.batch_size)))
        logger.logger.debug('[*] ave_accuracy_all %s' %
                            np.array2string(acc_all_count / ((batch_id+1) * FLAGS.batch_size)))
        logger.logger.debug('[*] ave_accuracy_2   %s' %
                            np.array2string(acc_2_count / ((batch_id+1) * FLAGS.batch_size)))

    print("[*] Finish testing.")

    logger.logger.info('[*] ave_hamming_dist %s' %
                       np.array2string(ham_dist_count / ((batch_id+1) * FLAGS.batch_size)))
    logger.logger.info('[*] ave_accuracy_all %s' %
                       np.array2string(acc_all_count / ((batch_id+1) * FLAGS.batch_size)))
    logger.logger.info('[*] ave_accuracy_2   %s' %
                       np.array2string(acc_2_count / ((batch_id+1) * FLAGS.batch_size)))

    return


def get_model_fn():
    if "puzzle" in FLAGS.task:
        return models.JPSModel


def get_eval_fn():
    if "puzzle" in FLAGS.task:
        return eval_puzzle


def evaluation(local_save_dir, sess, logger):
    data_iterator, data_init_op, num_batch = datasets.data_loader(
        is_training=False)
    sess.run(data_init_op)
    data = data_iterator.get_next()

    data_split = [{} for _ in range(FLAGS.gpu_num)]
    for k, t in data.items():
        t_split = tf.split(t, FLAGS.gpu_num, axis=0)
        for i, t_small in enumerate(t_split):
            data_split[i][k] = t_small

    model_list = []
    for i in range(FLAGS.gpu_num):
        with tf.device('/gpu:%d' % i):
            model_fn = get_model_fn()
            model = model_fn(data_split[i], is_training=False)
            model_list.append(model)

    eval_fn = get_eval_fn()
    eval_fn(model_list, sess, num_batch, logger)

    return


def train(local_save_dir):
    log = os.path.join(local_save_dir, 'log')
    if not os.path.exists(log):
        os.makedirs(log)
    logger = Logger(log + "/log", level=FLAGS.logger_level)

    with tf.device('cpu:0'):
        data_iterator, data_init_op, num_batch = datasets.data_loader(
            is_training=True)
        data = data_iterator.get_next()

        data_split = [{} for _ in range(FLAGS.gpu_num)]
        for k, t in data.items():
            t_split = tf.split(t, FLAGS.gpu_num, axis=0)
            for i, t_small in enumerate(t_split):
                data_split[i][k] = t_small

        optimizer = tf.train.MomentumOptimizer(FLAGS.base_lr, 0.9)

        grads = []
        display_losses = []
        for i in range(FLAGS.gpu_num):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%d' % i):
                    model_fn = get_model_fn()
                    model = model_fn(data_split[i],
                                     is_training=True)

                    grads_sub = []
                    for d in model.compute_gradients_losses:
                        grads_sub += optimizer.compute_gradients(
                            loss=d['value'], var_list=d['var_list'])
                    grads.append(grads_sub)

                display_losses += model.display_losses

        grads = utils.average_gradients(grads)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(grads)

        var_init_op = tf.group(tf.local_variables_initializer(),
                               tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=5)

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True))
        sess.run(var_init_op)
        sess.run(data_init_op)

        print(tf.trainable_variables())

        ckpt = os.path.join(local_save_dir, 'checkpoint')
        load_model_status, global_step = utils.load(sess, saver, ckpt)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // num_batch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            print("[*] Not find pretrained model!")

        start = time.time()
        for epoch_id in range(start_epoch, FLAGS.epochs):
            for batch_id in range(num_batch):
                _, losses_eval = sess.run([train_op, display_losses])

                end = time.time()

                losses_dict = {}
                for d in losses_eval:
                    if d['name'] in losses_dict.keys():
                        losses_dict[d['name']] += [d['value']]
                    else:
                        losses_dict[d['name']] = [d['value']]

                log = "Epoch: [%2d] [%4d/%4d] time: %s | " % (
                    epoch_id+1, batch_id+1, num_batch,
                    str(timedelta(seconds=end-start))[0:10])
                for k, v in losses_dict.items():
                    k = k.decode("utf-8")
                    log += "%s: %.6f " % (k, np.mean(v))
                logger.logger.info(log)
                iter_num += 1

            logger.logger.info(log)

            if np.mod(epoch_id + 1, FLAGS.save_every_epoch) == 0:
                utils.save(sess, saver, iter_num, ckpt)
            if np.mod(epoch_id + 1, FLAGS.eval_every_epoch) == 0:
                evaluation(local_save_dir, sess, logger)

        print("[*] Finish training.")


def test(local_save_dir):
    data_iterator, data_init_op, num_batch = datasets.data_loader(
        is_training=False)
    data = data_iterator.get_next()

    log = os.path.join(local_save_dir, 'log')
    if not os.path.exists(log):
        os.makedirs(log)
    logger = Logger(log + "/log", level=FLAGS.logger_level)

    with tf.device('/gpu:0'):
        model_fn = get_model_fn()
        model = model_fn(data, is_training=False)

    saver = tf.train.Saver()

    var_init_op = tf.group(tf.local_variables_initializer(),
                           tf.global_variables_initializer())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(var_init_op)
    sess.run(data_init_op)

    ckpt = os.path.join(local_save_dir, 'checkpoint')
    load_model_status, _ = utils.load(sess, saver, ckpt)
    if load_model_status:
        print("[*] Model restore success!")
    else:
        print("[*] Not find pretrained model!")

    eval_fn = get_eval_fn()
    eval_fn(model, sess, num_batch, logger)


def main(_):
    local_save_dir = os.path.join('/tmp/experiments/', FLAGS.experiment_name)

    if FLAGS.task == 'puzzle_train':
        train(local_save_dir)

    elif FLAGS.task == 'puzzle_eval':
        test(local_save_dir)

    else:
        print('[!]Unknown phase')
        exit(0)


if __name__ == '__main__':
    tf.app.run()
