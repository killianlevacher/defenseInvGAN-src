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

"""Testing blackbox Inv-GAN models. This module is based on MNIST tutorial
of cleverhans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import pickle
import logging
import os
import re
import sys

import numpy as np
import tensorflow as tf
from six.moves import xrange
from tensorflow.python.platform import flags


from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils import set_log_level, to_categorical
from cleverhans.utils_tf import model_train, model_eval, batch_eval


from models.gan_v2_art import InvertorDefenseGAN, gan_from_config
from utils.gan_defense_art import model_eval_gan
from utils.network_builder_art import model_a, model_e, model_f, DefenseWrapper
from utils.util_art import save_images_files, ensure_dir, load_config
from utils.reconstruction_art import Reconstructor
from utils.reconstruction_art import reconstruct_dataset

FLAGS = flags.FLAGS

# orig_ refers to original images and not reconstructed ones.
# To prepare these cache files run "python main.py --save_ds".
orig_data_path = {k: 'data/cache/{}_pkl'.format(k) for k in ['mnist']}
attack_config_dict = {'mnist': {'eps': 0.3, 'clip_min': 0}}


def prep_bbox(sess, images, labels, images_train, labels_train, images_test,
              labels_test, nb_epochs, batch_size, learning_rate, rng, gan=None,
              adv_training=False, cnn_arch=None):
    """Defines and trains a model that simulates the "remote"
    black-box oracle described in https://arxiv.org/abs/1602.02697.
    
    Args:
        sess: the TF session
        images: the input placeholder
        labels: the ouput placeholder
        images_train: the training data for the oracle
        labels_train: the training labels for the oracle
        images_test: the testing data for the oracle
        labels_test: the testing labels for the oracle
        nb_epochs: number of epochs to train model
        batch_size: size of training batches
        learning_rate: learning rate for training
        rng: numpy.random.RandomState
    
    Returns:
        model: The blackbox model function.
        predictions: The predictions tensor.
        accuracy: Accuracy of the model.
    """

    # Define TF model graph (for the black-box model).
    model = cnn_arch
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': 'classifiers/model/{}'.format("mnist"),
        'filename': 'model_{}'.format(FLAGS.bb_model)
    }
    eval_params = {'batch_size': batch_size}


    used_vars = model.get_params()
    pred_train = model.get_logits(images, dropout=True)
    pred_eval = model.get_logits(images)

    classifier_load_success = False
    if FLAGS.load_bb_model:
        try:
            path = tf.train.latest_checkpoint('classifiers/model/{}'.format("mnist"))
            saver = tf.train.Saver(var_list=used_vars)
            saver.restore(sess, path)
            print('[+] BB model loaded successfully ...')
            classifier_load_success = True
        except Exception as e:
            print('[-] Fail to load BB model ...')
            classifier_load_success = False

    if not classifier_load_success:
        print('[+] Training classifier model ...')
        model_train(sess, images, labels, pred_train, images_train, labels_train,
                args=train_params, rng=rng, predictions_adv=None, init_all=False, save=False)
    # Print out the accuracy on legitimate test data.
    accuracy = model_eval(
        sess, images, labels, pred_eval, images_test,
        labels_test, args=eval_params,
    )

    print('Test accuracy of black-box on legitimate test examples: ' + str(accuracy))

    return model, pred_eval, accuracy


def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              rng, substitute_model=None, dataset_name=None):
    """This function trains the substitute model as described in
        arxiv.org/abs/1602.02697
    Args:
        sess: TF session
        x: input TF placeholder
        y: output TF placeholder
        bbox_preds: output of black-box model predictions
        X_sub: initial substitute training data
        Y_sub: initial substitute training labels
        nb_classes: number of output classes
        nb_epochs_s: number of epochs to train substitute model
        batch_size: size of training batches
        learning_rate: learning rate for training
        data_aug: number of times substitute training data is augmented
        lmbda: lambda from arxiv.org/abs/1602.02697
        rng: numpy.random.RandomState instance
    
    Returns:
        model_sub: The substitute model function.
        preds_sub: The substitute prediction tensor.
    """

    model_sub = substitute_model
    used_vars = model_sub.get_params()

    if FLAGS.load_sub_model:
        try:
            path = tf.train.latest_checkpoint('classifiers/sub_model/{}'.format(dataset_name))
            saver = tf.train.Saver(var_list=used_vars)
            saver.restore(sess, path)
            print('[+] Sub model loaded successfully ...')

            pred_eval = model_sub.get_logits(x)
            return model_sub, pred_eval

        except:
            pass

    pred_train = model_sub.get_logits(x, dropout=True)
    pred_eval = model_sub.get_logits(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow.
    grads = jacobian_graph(pred_eval, x, nb_classes)

    train_params = {
        'nb_epochs': nb_epochs_s,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': 'classifiers/sub_model/{}'.format(dataset_name),
        'filename': 'model_{}'.format(FLAGS.sub_model)
    }

    # Train the substitute and augment dataset alternatively.
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        model_train(sess, x, y, pred_train, X_sub, convert_to_onehot(Y_sub),
                    init_all=False, args=train_params,
                    rng=rng, save=True)

        # If we are not at last substitute training iteration, augment dataset.
        if rho < data_aug - 1:

            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation.
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads, lmbda)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box.
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub) / 2):]
            eval_params = {'batch_size': batch_size}

            # To initialize the local variables of Defense-GAN.
            sess.run(tf.local_variables_initializer())

            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params)[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model.
            Y_sub[int(len(X_sub) / 2):] = np.argmax(bbox_val, axis=1)

    return model_sub, pred_eval


def convert_to_onehot(ys):
    """Converts the labels to one-hot vectors."""
    max_y = int(np.max(ys))
    y_one_hat = np.zeros([len(ys), max_y + 1], np.float32)
    for (i, y) in enumerate(ys):
        y_one_hat[i, int(y)] = 1.0
    return y_one_hat




def get_train_test(data_path, test_on_dev=True, model=None,
                   orig_data=False, max_num=-1):
    """Loads the datasets.
    Args:
        data_path: The path that contains train,dev,[test] directories
        test_on_dev: Test on the development set
        model: An instance of `GAN`.
        orig_data: `True` for loading original data, `False` to load the
            reconstructed images.
    
    Returns:
        train_images: Training images.
        train_labels: Training labels.
        test_images: Testing images.
        test_labels: Testing labels.
    """

    data_dict = None
    if model and not orig_data:
        data_dict = reconstruct_dataset(gan_model=model, max_num_load=max_num)

    def get_images_labels_from_pickle(data_path, split):
        data_path = os.path.join(data_path, split, 'feats.pkl')
        could_load = False
        try:
            if os.path.exists(data_path):
                with open(data_path,'rb') as f:
                    train_images_gan = pickle.load(f)
                    train_labels_gan = pickle.load(f)
                could_load = True
            else:
                print('[!] Run python train.py --cfg <path-to-cfg> --save_ds to prepare the dataset cache files.')
                exit(1)

        except Exception as e:
            print('[!] Found feats.pkl but could not load it because {}'.format(str(e)))

        if data_dict is not None:
            # 'data_dict is not None' implies 'orig_data is False'
            # it doesn't matter whether original data is loaded
            train_images_gan, train_labels_gan, train_images_orig = data_dict[split]
        elif could_load is False:
            print('[!] Could not load feats.pkl. Instead, please use reconstructed data.')
            exit(1)

        return train_images_gan, convert_to_onehot(train_labels_gan)

    train_images, train_lables = get_images_labels_from_pickle(data_path, 'train')
    test_split = 'dev' if test_on_dev else 'test'
    test_images, test_labels = get_images_labels_from_pickle(data_path, test_split)

    return train_images, train_lables, test_images, test_labels


def get_cached_gan_data(gan, test_on_dev, orig_data_flag=None):
    """Fetches the dataset of a GAN model.
    
    Args:
        gan: The GAN model.
        test_on_dev: `True` for loading the dev set instead of the test set.
        orig_data_flag: `True` for loading the original images not the 
            reconstructions.
    Returns:
        train_images: Training images.
        train_labels: Training labels.
        test_images: Testing images.
        test_labels: Testing labels.
    """
    FLAGS = flags.FLAGS
    if orig_data_flag is None:
        if not FLAGS.train_on_recs or FLAGS.defense_type != 'defense_gan':
            orig_data_flag = True
        else:
            orig_data_flag = False


    train_images, train_labels, test_images, test_labels = \
        get_train_test(
            orig_data_path["mnist"], test_on_dev=test_on_dev,
            model=gan, orig_data=orig_data_flag, max_num=FLAGS.num_train)
    return train_images, train_labels, test_images, test_labels


def blackbox(gan, rec_data_path=None, batch_size=128,
             learning_rate=0.001, nb_epochs=10, holdout=150, data_aug=6,
             nb_epochs_s=10, lmbda=0.1, online_training=False,
             train_on_recs=False, test_on_dev=False,
             defense_type='none'):
    """MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697
    
    Args:
        train_start: index of first training set example
        train_end: index of last training set example
        test_start: index of first test set example
        test_end: index of last test set example
        defense_type: Type of defense against blackbox attacks
    
    Returns:
        a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """
    FLAGS = flags.FLAGS

    # Set logging level to see debug information.
    set_log_level(logging.WARNING)

    # Dictionary used to keep track and return key accuracies.
    accuracies = {}

    # Create TF session.
    adv_training = False
    if defense_type:
        if defense_type == 'defense_gan' and gan:
            sess = gan.sess
            gan_defense_flag = True
        else:
            gan_defense_flag = False
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
        if 'adv_tr' in defense_type:
            adv_training = True
    else:
        gan_defense_flag = False
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    train_images, train_labels, test_images, test_labels = \
        get_cached_gan_data(gan, test_on_dev, orig_data_flag=True)

    x_shape, classes = list(train_images.shape[1:]), train_labels.shape[1]
    nb_classes = classes

    type_to_models = {
        'A': model_a, 'E': model_e, 'F': model_f
    }

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        bb_model = type_to_models[FLAGS.bb_model](
            input_shape=[None] + x_shape, nb_classes=train_labels.shape[1],
        )
    with tf.variable_scope("Substitute", reuse=tf.AUTO_REUSE):
        sub_model = type_to_models[FLAGS.sub_model](
            input_shape=[None] + x_shape, nb_classes=train_labels.shape[1],
        )

    if FLAGS.debug:
        train_images = train_images[:20 * batch_size]
        train_labels = train_labels[:20 * batch_size]
        debug_dir = os.path.join('debug', 'blackbox', FLAGS.debug_dir)
        ensure_dir(debug_dir)
        x_debug_test = test_images[:batch_size]

    # Initialize substitute training set reserved for adversary
    images_sub = test_images[:holdout]
    labels_sub = np.argmax(test_labels[:holdout], axis=1)

    print(labels_sub)

    # Redefine test set as remaining samples unavailable to adversaries
    if FLAGS.num_tests > 0:
        test_images = test_images[:FLAGS.num_tests]
        test_labels = test_labels[:FLAGS.num_tests]

    test_images = test_images[holdout:]
    test_labels = test_labels[holdout:]

    # Define input and output TF placeholders

    # TODO maybe that should be put back but I don't understand where it's set
    # if FLAGS.image_dim[0] == 3:
    #     FLAGS.image_dim = [FLAGS.image_dim[1], FLAGS.image_dim[2],
    #                        FLAGS.image_dim[0]]

    images_tensor = tf.placeholder(tf.float32, shape=[None] + x_shape)
    labels_tensor = tf.placeholder(tf.float32, shape=(None, classes))

    rng = np.random.RandomState([11, 24, 1990])

    train_images_bb, train_labels_bb, test_images_bb, test_labels_bb = \
        train_images, train_labels, test_images, \
        test_labels

    cur_gan = gan
    if FLAGS.debug:
        train_images_bb = train_images_bb[:20 * batch_size]
        train_labels_bb = train_labels_bb[:20 * batch_size]

    # Prepare the black_box model.
    prep_bbox_out = prep_bbox(
        sess, images_tensor, labels_tensor, train_images_bb,
        train_labels_bb, test_images_bb, test_labels_bb, nb_epochs,
        batch_size, learning_rate, rng=rng, gan=cur_gan,
        adv_training=adv_training,
        cnn_arch=bb_model)

    #accuracies['bbox'] is the legitimate accuracy
    model, bbox_preds, accuracies['bbox'] = prep_bbox_out



    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    reconstructor = get_reconstructor(gan)
    recon_tensors, _ = reconstructor.reconstruct(images_tensor, batch_size=batch_size, reconstructor_id=2)

    model_sub, preds_sub = train_sub(
        sess, images_tensor, labels_tensor,
        model.get_logits(recon_tensors), images_sub,
        labels_sub,
        nb_classes, nb_epochs_s, batch_size,
        learning_rate, data_aug, lmbda, rng=rng,
        substitute_model=sub_model, dataset_name=gan.dataset_name
    )

    accuracies['sub'] = 0

    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    eps = attack_config_dict[gan.dataset_name]['eps']
    min_val = attack_config_dict[gan.dataset_name]['clip_min']

    fgsm_par = {
        'eps': eps, 'ord': np.inf, 'clip_min': min_val, 'clip_max': 1.
    }

    fgsm = FastGradientMethod(model_sub, sess=sess)

    # Craft adversarial examples using the substitute.
    eval_params = {'batch_size': batch_size}
    x_adv_sub = fgsm.generate(images_tensor, **fgsm_par)

    if FLAGS.debug and gan is not None:  # To see some qualitative results.
        recon_tensors, _ = reconstructor.reconstruct(x_adv_sub, batch_size=batch_size, reconstructor_id=2)
        x_rec_orig, _ = reconstructor.reconstruct(images_tensor, batch_size=batch_size, reconstructor_id=3)

        x_adv_sub_val = sess.run(x_adv_sub, feed_dict={images_tensor: x_debug_test})
        x_rec_debug_val = sess.run(recon_tensors, feed_dict={images_tensor: x_debug_test})
        x_rec_orig_val = sess.run(x_rec_orig, feed_dict={images_tensor: x_debug_test})
        #sess.run(tf.local_variables_initializer())
        #x_rec_debug_val, x_rec_orig_val = sess.run([reconstructed_tensors, x_rec_orig], feed_dict={images_tensor: x_debug_test})

        save_images_files(x_adv_sub_val, output_dir=debug_dir,
                          postfix='adv')

        postfix = 'gen_rec'
        save_images_files(x_rec_debug_val, output_dir=debug_dir,
                          postfix=postfix)
        save_images_files(x_debug_test, output_dir=debug_dir,
                          postfix='orig')
        save_images_files(x_rec_orig_val, output_dir=debug_dir, postfix='orig_rec')

    if gan_defense_flag:
        num_dims = len(images_tensor.get_shape())
        avg_inds = list(range(1, num_dims))

        recons_adv, zs = reconstructor.reconstruct(x_adv_sub, batch_size=batch_size)

        diff_op = tf.reduce_mean(tf.square(x_adv_sub - recons_adv), axis=avg_inds)
        z_norm = tf.reduce_sum(tf.square(zs), axis=1)

        acc_adv, diffs_mean, roc_info_adv = model_eval_gan(sess, images_tensor, labels_tensor,
                              predictions=model.get_logits(recons_adv),
                              test_images=test_images, test_labels=test_labels,
                              args=eval_params, diff_op=diff_op,
                              z_norm=z_norm, recons_adv=recons_adv, adv_x=x_adv_sub, debug=False)

        # reconstruction on clean images
        recons_clean, zs = reconstructor.reconstruct(images_tensor, batch_size=batch_size)

        diff_op = tf.reduce_mean(tf.square(images_tensor - recons_clean), axis=avg_inds)
        z_norm = tf.reduce_sum(tf.square(zs), axis=1)

        acc_rec, diffs_mean_rec, roc_info_rec = model_eval_gan(
            sess, images_tensor, labels_tensor, model.get_logits(recons_clean), None,
            test_images=test_images, test_labels=test_labels, args=eval_params, diff_op=diff_op,
            z_norm=z_norm, recons_adv=recons_clean, adv_x=images_tensor, debug=False)

        print('Evaluation accuracy with reconstruction: {}'.format(acc_rec))
        print('Test accuracy of oracle on cleaned images : {}'.format(acc_adv))
        print('Test accuracy of black-box on non adversarial test examples: ' + str(accuracies['bbox']))
        return {'acc_adv': acc_adv,
                'acc_rec': acc_rec,
                'roc_info_adv': roc_info_adv,
                'roc_info_rec': roc_info_rec}

    else:
        acc_adv = model_eval(sess, images_tensor, labels_tensor,
                              model.get_logits(x_adv_sub), test_images,
                              test_labels,
                              args=eval_params)
        print('Test accuracy of oracle on adversarial examples generated '
              'using the substitute: ' + str(acc_adv))
        print('Test accuracy of black-box on non adversarial test examples: ' + str(accuracies['bbox']))
        return {'acc_adv': acc_adv,
                'acc_rec': 0,
                'roc_info_adv': None,
                'roc_info_rec': None}



def get_reconstructor(gan):
    if isinstance(gan, InvertorDefenseGAN):
        return Reconstructor(gan)
    else:
        return gan


def _get_results_dir_filename(gan):
    result_file_name = 'sub={:d}_eps={:.2f}.txt'.format(FLAGS.data_aug,
                                                        FLAGS.fgsm_eps)

    # results_dir = os.path.join('results', '{}_{}'.format(
    #     FLAGS.defense_type, FLAGS.dataset_name))

    results_dir = os.path.join('results', 'blackbox_{}_{}'.format(FLAGS.defense_type, "mnist"))

    if FLAGS.defense_type == 'defense_gan':
        results_dir = gan.checkpoint_dir.replace('output', 'results')
        result_file_name = \
            'RR={:d}_LR={:.4f}_Iter={:d}_sub={:d}.txt'.format(
                gan.rec_rr,
                gan.rec_lr,
                gan.rec_iters,
                FLAGS.data_aug)

        if not FLAGS.train_on_recs:
            result_file_name = 'orig_' + result_file_name
    elif FLAGS.defense_type == 'adv_tr':
        result_file_name = 'sub={:d}_trEps={:.2f}_eps={:.2f}.txt'.format(
            FLAGS.data_aug, FLAGS.fgsm_eps_tr,
            FLAGS.fgsm_eps)
    if FLAGS.num_tests > -1:
        result_file_name = 'numtest={}_'.format(
            FLAGS.num_tests) + result_file_name

    if FLAGS.num_train > -1:
        result_file_name = 'numtrain={}_'.format(
            FLAGS.num_train) + result_file_name

    result_file_name = 'bbModel={}_subModel={}_'.format(FLAGS.bb_model,
                                                        FLAGS.sub_model) \
                       + result_file_name
    return results_dir, result_file_name


def main(cfg, argv=None):
    FLAGS = tf.app.flags.FLAGS

    tf.set_random_seed(11241990)
    np.random.seed(11241990)

    gan = None
    # Setting test time reconstruction hyper parameters.
    # [tr_rr, tr_lr, tr_iters] = [FLAGS.rec_rr, FLAGS.rec_lr, FLAGS.rec_iters]
    [tr_rr, tr_lr, tr_iters] = [cfg["REC_RR"], cfg["REC_LR"], cfg["REC_ITERS"]]

    if FLAGS.defense_type.lower() != 'none':
        if FLAGS.defense_type == 'defense_gan':
            gan = gan_from_config(cfg, True)

            gan.load_model()

            # extract hyper parameters from reconstruction path.
            if FLAGS.rec_path is not None:
                train_param_re = re.compile('recs_rr(.*)_lr(.*)_iters(.*)')
                [tr_rr, tr_lr, tr_iters] = \
                    train_param_re.findall(FLAGS.rec_path)[0]
                gan.rec_rr = int(tr_rr)
                gan.rec_lr = float(tr_lr)
                gan.rec_iters = int(tr_iters)
            else:
                assert FLAGS.online_training or not FLAGS.train_on_recs


    if FLAGS.override:
        gan.rec_rr = int(tr_rr)
        gan.rec_lr = float(tr_lr)
        gan.rec_iters = int(tr_iters)

    # Setting the reuslts directory
    results_dir, result_file_name = _get_results_dir_filename(gan)

    # Result file name. The counter makes sure we are not overwriting the
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

    accuracies = blackbox(gan, rec_data_path=FLAGS.rec_path,
                          batch_size=cfg["BATCH_SIZE"],
                          learning_rate=FLAGS.learning_rate,
                          nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                          data_aug=FLAGS.data_aug,
                          nb_epochs_s=FLAGS.nb_epochs_s,
                          lmbda=FLAGS.lmbda,
                          online_training=FLAGS.online_training,
                          train_on_recs=FLAGS.train_on_recs,
                          defense_type=cfg["TYPE"])


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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Note: The load_config() call will convert all the parameters that are defined in
    # experiments/config files into FLAGS.param_name and can be passed in from command line.
    # arguments : python blackbox.py --cfg <config_path> --<param_name> <param_value>
    cfg = load_config(args.cfg)
    flags = tf.app.flags

    # ISSUE: the classifiers provided by authors only contain model A - other models need to be trained
    flags.DEFINE_string("bb_model", 'A',
                        "The architecture of the classifier model.")
    flags.DEFINE_string("sub_model", 'A', "The architecture of the substitute model.")
    flags.DEFINE_boolean("load_bb_model", True, "True for loading from saved bb models [False]")
    flags.DEFINE_boolean("load_sub_model", True, "True for loading from saved sub models [False]")
    flags.DEFINE_string("defense_type", "defense_gan", "Type of defense " "[defense_gan|adv_tr|none]")


    ##############

    flags.DEFINE_integer('nb_classes', 10, 'Number of classes.')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training '
                                               'the black-box model.')
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train the '
                                          'blackbox model.')
    flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary.')
    flags.DEFINE_integer('data_aug', 6, 'Number of substitute data augmentations.')
    flags.DEFINE_integer('nb_epochs_s', 10, 'Training epochs for substitute.')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')
    flags.DEFINE_float('fgsm_eps', 0.3, 'FGSM epsilon.')
    flags.DEFINE_float('fgsm_eps_tr', 0.15, 'FGSM epsilon for adversarial '
                                            'training.')
    flags.DEFINE_string('rec_path', None, 'Path to Defense-GAN '
                                          'reconstructions.')
    flags.DEFINE_integer('num_tests', -1, 'Number of test samples.')
    flags.DEFINE_integer('random_test_iter', -1,
                         'Number of random sampling for testing the '
                         'classifier.')
    flags.DEFINE_boolean("online_training", False,
                         'Train the base classifier based on online '
                         'reconstructions from Defense-GAN, as opposed to '
                         'using the cached reconstructions.')


    flags.DEFINE_string("results_dir", "blackbox", "The path to results.")
    flags.DEFINE_boolean("train_on_recs", False,
                         "Train the black-box model on Defense-GAN "
                         "reconstructions.")
    flags.DEFINE_integer('num_train', -1, 'Number of training samples for '
                                          'the black-box model.')




    flags.DEFINE_string("debug_dir", "temp", "Directory for debug outputs.")
    flags.DEFINE_boolean("debug", False, "True for saving reconstructions [False]")
    flags.DEFINE_boolean("override", None, "Overrides the test hyperparams.")

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)
