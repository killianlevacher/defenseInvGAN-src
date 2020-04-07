# Copyright 2019 The Inv-GAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Testing white-box attacks Inv-GAN models. This module is based on MNIST
tutorial of cleverhans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import _init_paths

import argparse
import pickle
import logging
import os
import sys

import numpy as np
import tensorflow as tf

from classifiers.cifar_model import Model
from blackbox import get_cached_gan_data, get_reconstructor
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks.bpda import BPDAL2
from utils.attack import MadryEtAl
from cleverhans.utils import set_log_level, AccuracyReport
from cleverhans.utils_tf import model_train, model_eval
from models.gan_v2 import InvertorDefenseGAN
from models.dataset_networks import get_generator_fn
from models.gan_v2 import DefenseGANv2, InvertorDefenseGAN
from utils.config import load_config
from utils.gan_defense import model_eval_gan
from utils.misc import ensure_dir
from utils.network_builder import model_a

################# PARAMS
# DEFENSE_TYPE = "none"
# DEFENSE_TYPE = "defense_gan"
# DEBUG=True
# DETECT_IMAGE = False
# LOAD_CLASSIFIER = True
#
# FLAGSattack_type = "fgsm"
# # flags.DEFINE_integer("attack_iters", 100, 'Number of iterations for cw/pgd attack.')
#
#
# FLAGSfgsm_eps = 0.3
#
# FLAGSdebug_dir = "temp"
# FLAGSdetect_image = False
#
# FLAGSlearning_rate = 0.001
# FLAGSlmbda= 0.1
#
# FLAGSmodel = "A"
#
# FLAGSnb_classes = 10
# FLAGSnb_epochs = 10
# FLAGSnum_tests = -1
# FLAGSnum_train = -1
#
# FLAGSoverride =  False
# FLAGSonline_training = False
#
# FLAGSrec_path= None
# FLAGSrandom_test_iter = -1
# FLAGSresults_dir= "whitebox"
# FLAGSsame_init =  False
# FLAGStrain_on_recs = False

#################
orig_data_paths = {k: 'data/cache/{}_pkl'.format(k) for k in ['mnist', 'f-mnist', 'cifar-10']}
attack_config_dict = {'mnist': {'eps': 0.3, 'clip_min': 0},
                      'f-mnist': {'eps': 0.3, 'clip_min': 0},
                      'cifar-10': {'eps': 8*2 / 255.0, 'clip_min': -1},
                      'celeba': {'eps': 8*2 / 255.0, 'clip_min': -1}
                      }


def get_diff_op(classifier, x1, x2, use_image=False):

    if use_image:
        f1 = x1
        f2 = x2
    else:
        f1 = classifier.extract_feature(x1)
        f2 = classifier.extract_feature(x2)

    num_dims = len(f1.get_shape())
    avg_inds = list(range(1, num_dims))

    return tf.reduce_mean(tf.square(f1 - f2), axis=avg_inds)


def whitebox(gan, rec_data_path=None, batch_size=128, learning_rate=0.001,
             nb_epochs=10, eps=0.3, online_training=False,
             test_on_dev=False, attack_type='fgsm', defense_type='gan',
             num_tests=-1, num_train=-1, cfg=None):
    """Based on MNIST tutorial from cleverhans.
    
    Args:
         gan: A `GAN` model.
         rec_data_path: A string to the directory.
         batch_size: The size of the batch.
         learning_rate: The learning rate for training the target models.
         nb_epochs: Number of epochs for training the target model.
         eps: The epsilon of FGSM.
         online_training: Training Defense-GAN with online reconstruction. The
            faster but less accurate way is to reconstruct the dataset once and use
            it to train the target models with:
            `python train.py --cfg <path-to-model> --save_recs`
         attack_type: Type of the white-box attack. It can be `fgsm`,
            `rand+fgsm`, or `cw`.
         defense_type: String representing the type of attack. Can be `none`,
            `defense_gan`, or `adv_tr`.
    """
    
    FLAGS = tf.flags.FLAGS
    rng = np.random.RandomState([11, 24, 1990])

    # Set logging level to see debug information.
    set_log_level(logging.WARNING)

    ### Attack paramters
    eps = attack_config_dict[gan.dataset_name]['eps']
    min_val = attack_config_dict[gan.dataset_name]['clip_min']

    train_images, train_labels, test_images, test_labels = get_cached_gan_data(gan, test_on_dev, orig_data_flag=True)

    SUB_BATCH_SIZE = batch_size
    test_images = test_images[:SUB_BATCH_SIZE]
    test_labels = test_labels[:SUB_BATCH_SIZE]

    sess = gan.sess

    # Classifier is trained on either original data or reconstructed data.
    # During testing, the input image will be reconstructed by GAN.
    # Therefore, we use rec_test_images as input to the classifier.
    # When evaluating defense_gan with attack, input should be test_images.

    x_shape = [None] + list(train_images.shape[1:])
    images_pl = tf.placeholder(tf.float32, shape=[None] + list(train_images.shape[1:]))
    labels_pl = tf.placeholder(tf.float32, shape=[None] + [train_labels.shape[1]])

    # Creating classificaion model
    images_pl_transformed = images_pl
    model = model_a

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        model = model(input_shape=x_shape, nb_classes=train_labels.shape[1])

    used_vars = model.get_params()
    preds_train = model.get_logits(images_pl_transformed, dropout=True)
    preds_eval = model.get_logits(images_pl_transformed)

    report = AccuracyReport()

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test
        # examples.
        eval_params = {'batch_size': batch_size}
        acc = model_eval(
            sess, images_pl, labels_pl, preds_eval, test_images, test_labels, args=eval_params)
        report.clean_train_clean_eval = acc
        print('Test accuracy: %0.4f' % acc)

    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': 'classifiers/model/{}'.format(gan.dataset_name),
        'filename': 'model_a'
    }

    preds_adv = None

    classifier_load_success = False
    if FLAGS.load_classifier:
        try:
            path = tf.train.latest_checkpoint('classifiers/model/{}'.format(gan.dataset_name))
            saver = tf.train.Saver(var_list=used_vars)
            saver.restore(sess, path)
            print('[+] Classifier loaded successfully ...')
            classifier_load_success = True
        except:
            print('[-] Cannot load classifier ...')
            classifier_load_success = False


    # Calculate training error.
    eval_params = {'batch_size': batch_size}

    # Evaluate trained model
    #train_acc = model_eval(sess, images_pl, labels_pl, preds_eval, train_images, train_labels,
    #                       args=eval_params)
    # print('[#] Train acc: {}'.format(train_acc))

    eval_acc = model_eval(sess, images_pl, labels_pl, preds_eval, test_images, test_labels,
                          args=eval_params)
    print('[#] Non Adversarial Eval accuracy: {}'.format(eval_acc))

    reconstructor = get_reconstructor(gan)

    if attack_type is None:
        return eval_acc, 0, None

    attack_params = {'eps': eps, 'ord': np.inf, 'clip_min': min_val, 'clip_max': 1.}
    attack_obj = FastGradientMethod(model, sess=sess)

    adv_x = attack_obj.generate(images_pl_transformed, **attack_params)

    if cfg["TYPE"] == 'defense_gan':

        recons_adv, zs = reconstructor.reconstruct(adv_x, batch_size=batch_size, reconstructor_id=123)

        preds_adv = model.get_logits(recons_adv)

        sess.run(tf.local_variables_initializer())

        diff_op = get_diff_op(model, adv_x, recons_adv, FLAGS.detect_image)
        z_norm = tf.reduce_sum(tf.square(zs), axis=1)

        acc_adv, diffs_mean, roc_info_adv = model_eval_gan(
            sess, images_pl, labels_pl, preds_adv, None,
            test_images=test_images, test_labels=test_labels, args=eval_params, diff_op=diff_op,
            z_norm=z_norm, recons_adv=recons_adv, adv_x=adv_x, debug=FLAGS.debug, vis_dir=_get_vis_dir(gan, FLAGS.attack_type))

        # reconstruction on clean images
        recons_clean, zs = reconstructor.reconstruct(images_pl_transformed, batch_size=batch_size)
        preds_eval = model.get_logits(recons_clean)

        sess.run(tf.local_variables_initializer())

        diff_op = get_diff_op(model, images_pl_transformed, recons_clean, FLAGS.detect_image)
        z_norm = tf.reduce_sum(tf.square(zs), axis=1)

        acc_rec, diffs_mean_rec, roc_info_rec = model_eval_gan(
            sess, images_pl, labels_pl, preds_eval, None,
            test_images=test_images, test_labels=test_labels, args=eval_params, diff_op=diff_op,
            z_norm=z_norm, recons_adv=recons_clean, adv_x=images_pl, debug=FLAGS.debug, vis_dir=_get_vis_dir(gan, 'clean'))

        # print('Training accuracy: {}'.format(train_acc))
        print('Non Adversarial Eval accuracy: {}'.format(eval_acc))
        print('Adversarial eval accuracy: %0.4f' % acc_adv)
        print('Evaluation accuracy with defence: {}'.format(acc_rec))

        return {'acc_adv': acc_adv,
                'acc_rec': acc_rec,
                'roc_info_adv': roc_info_adv,
                'roc_info_rec': roc_info_rec}
    else:
        preds_adv = model.get_logits(adv_x)
        sess.run(tf.local_variables_initializer())
        acc_adv = model_eval(sess, images_pl, labels_pl, preds_adv, test_images, test_labels,
                             args=eval_params)
        print('Test accuracy on adversarial examples with No defense: %0.4f\n' % acc_adv)

        return {'acc_adv': acc_adv,
                'acc_rec': 0,
                'roc_info_adv': None,
                'roc_info_rec': None}


import re

def gan_from_config(cfg, test_mode):
# from config.py
    if cfg['TYPE'] == 'v2':
        gan = DefenseGANv2(
            get_generator_fn(cfg['DATASET_NAME'], cfg['USE_RESBLOCK']), cfg=cfg,
            test_mode=test_mode,
        )
    elif cfg['TYPE'] == 'inv':
        gan = InvertorDefenseGAN(
            get_generator_fn(cfg['DATASET_NAME'], cfg['USE_RESBLOCK']), cfg=cfg,
            test_mode=test_mode,
        )
    else:
        raise Exception("dummy")
        # gan = DefenseGANBase(cfg=cfg, test_mode=test_mode)
    return gan

def main(cfg, argv=None):
    FLAGS = tf.app.flags.FLAGS

    tf.set_random_seed(11241990)
    np.random.seed(11241990)

    # Setting test time reconstruction hyper parameters.
    # [tr_rr, tr_lr, tr_iters] = [FLAGS.rec_rr, FLAGS.rec_lr, FLAGS.rec_iters]
    [tr_rr, tr_lr, tr_iters] = [cfg["REC_RR"], cfg["REC_LR"], cfg["REC_ITERS"]]

    gan = None
    if cfg['TYPE'].lower() != 'none':
        if cfg['TYPE'] == 'defense_gan':
            gan = gan_from_config(cfg, True)

            gan.load_model()

            # Extract hyperparameters from reconstruction path.

            assert FLAGS.online_training or not FLAGS.train_on_recs

    if gan is None:
        # TODO NOT USED 2 - USED FOR NO DEFENCE
        gan = gan_from_config(cfg, True)
        gan.load_model()
        # gan = DefenseGANBase(cfg=cfg, test_mode=True)

    # Setting the results directory.
    results_dir, result_file_name = _get_results_dir_filename(cfg, gan)

    # Result file name. The counter ensures we are not overwriting the
    # results.
    counter = 0
    temp_fp = str(counter) + '_' + result_file_name
    results_dir = os.path.join(results_dir, FLAGS.results_dir)
    temp_final_fp = os.path.join(results_dir, temp_fp)
    while os.path.exists(temp_final_fp):
        counter += 1
        temp_fp = str(counter) + '_' + result_file_name
        temp_final_fp = os.path.join(results_dir, temp_fp)
    result_file_name = temp_fp
    sub_result_path = os.path.join(results_dir, result_file_name)


    accuracies = whitebox(
        gan, rec_data_path=FLAGS.rec_path,
        batch_size=cfg["BATCH_SIZE"],
        learning_rate=FLAGS.learning_rate,
        nb_epochs=FLAGS.nb_epochs,
        eps=FLAGS.fgsm_eps,
        online_training=FLAGS.online_training,
        defense_type=cfg["TYPE"],
        num_tests=FLAGS.num_tests,
        attack_type=FLAGS.attack_type,
        num_train=FLAGS.num_train,
        cfg=cfg
    )

    ensure_dir(results_dir)

    with open(sub_result_path, 'a') as f:
        f.writelines([str(accuracies['acc_adv']) + ' ' + str(accuracies['acc_rec']) + '\n'])
        print('[*] saved accuracy in {}'.format(sub_result_path))

    if accuracies['roc_info_adv']:  # For attack detection.
        pkl_result_path = sub_result_path.replace('.txt', '_roc.pkl')
        with open(pkl_result_path, 'wb') as f:
            pickle.dump(accuracies['roc_info_adv'], f)
            # cPickle.dump(accuracies['roc_info_adv'], f, cPickle.HIGHEST_PROTOCOL)
            print('[*] saved roc_info in {}'.format(pkl_result_path))

    if accuracies['roc_info_rec']:  # For attack detection.
        pkl_result_path = sub_result_path.replace('.txt', '_roc_clean.pkl')
        with open(pkl_result_path, 'wb') as f:
            pickle.dump(accuracies['roc_info_rec'], f)
            # cPickle.dump(accuracies['roc_info_rec'], f, cPickle.HIGHEST_PROTOCOL)
            print('[*] saved roc_info_clean in {}'.format(pkl_result_path))


def _get_results_dir_filename(cfg, gan):
    FLAGS = tf.flags.FLAGS

    results_dir = os.path.join('results', 'whitebox_{}_{}'.format(
        cfg['TYPE'], cfg["DATASET_NAME"]))

    if cfg['TYPE'] == 'defense_gan':
        results_dir = gan.checkpoint_dir.replace('output', 'results')
        result_file_name = \
            'Iter={}_RR={:d}_LR={:.4f}_defense_gan'.format(
                gan.rec_iters,
                gan.rec_rr,
                gan.rec_lr,
                FLAGS.attack_type,
            )

        if not FLAGS.train_on_recs:
            result_file_name = 'orig_' + result_file_name
    else:
        result_file_name = 'nodefense_'
    if FLAGS.num_tests > -1:
        result_file_name = 'numtest={}_'.format(
            FLAGS.num_tests) + result_file_name

    if FLAGS.num_train > -1:
        result_file_name = 'numtrain={}_'.format(
            FLAGS.num_train) + result_file_name

    if FLAGS.detect_image:
        result_file_name = 'det_image_' + result_file_name

    result_file_name = 'model={}_'.format(FLAGS.model) + result_file_name
    result_file_name += 'attack={}.txt'.format(FLAGS.attack_type)
    return results_dir, result_file_name


def _get_vis_dir(gan, attack_type):
    vis_dir = gan.checkpoint_dir.replace('output', 'vis')
    vis_dir = os.path.join(vis_dir, attack_type)
    return vis_dir


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Note: The load_config() call will convert all the parameters that are defined in
    # experiments/config files into FLAGS.param_name and can be passed in from command line.
    # arguments : python whitebox.py --cfg <config_path> --<param_name> <param_value>

    cfg = load_config(args.cfg)
    flags = tf.app.flags

    # flags.DEFINE_integer("num_train", -1, 'Number of training data to load.')
    flags = tf.app.flags

    flags.DEFINE_integer('nb_classes', 10, 'Number of classes.')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training.')
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model.')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697.')
    flags.DEFINE_float('fgsm_eps', 0.3, 'FGSM epsilon.')
    flags.DEFINE_string('rec_path', None, 'Path to reconstructions.')
    flags.DEFINE_integer('num_tests', -1, 'Number of test samples.')
    flags.DEFINE_integer('random_test_iter', -1,
                         'Number of random sampling for testing the classifier.')
    flags.DEFINE_boolean("online_training", False,
                         "Train the base classifier on reconstructions.")
    flags.DEFINE_string("defense_type", "none", "Type of defense [none|defense_gan|adv_tr]")
    flags.DEFINE_string("attack_type", "none", "Type of attack [fgsm|cw|bpda]")
    flags.DEFINE_integer("attack_iters", 100, 'Number of iterations for cw/pgd attack.')
    flags.DEFINE_integer("search_steps", 4, 'Number of binary search steps.')
    flags.DEFINE_string("results_dir", "result_subdir", "The final subdirectory of the results.")
    flags.DEFINE_boolean("same_init", False, "Same initialization for z_hats.")
    flags.DEFINE_string("model", "F", "The classifier model.")
    flags.DEFINE_string("debug_dir", "temp", "The debug directory.")
    flags.DEFINE_integer("num_train", -1, 'Number of training data to load.')
    flags.DEFINE_boolean("debug", False, "True for saving reconstructions [False]")
    flags.DEFINE_boolean("load_classifier", False, "True for loading from saved classifier models [False]")
    flags.DEFINE_boolean("detect_image", False, "True for detection using image data [False]")
    flags.DEFINE_boolean("override", False, "Overriding the config values of reconstruction "
                                            "hyperparameters. It has to be true if either "
                                            "`--rec_rr`, `--rec_lr`, or `--rec_iters` is passed "
                                            "from command line.")
    flags.DEFINE_boolean("train_on_recs", False,
                         "Train the classifier on the reconstructed samples "
                         "using Defense-GAN.")

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)
