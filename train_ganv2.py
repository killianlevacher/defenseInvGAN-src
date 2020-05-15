# Copyright 2018 The Defense-GAN Authors. All Rights Reserved.
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
# =============================================================================

"""The main class for training GANs."""

import argparse
import sys

import tensorflow as tf
# from models.cgan import DefenseCGAN
from models.gan_v2_art import DefenseGANv2
from utils.config import load_config
from utils.reconstruction import reconstruct_dataset, save_ds, encoder_reconstruct
from utils.metrics import compute_inception_score, save_mse
from utils.util_art import get_generator_fn
import numpy as np

#Killian added
cfg = {'TYPE': 'inv',
       'MODE': 'hingegan',
       # 'BATCH_SIZE': batch_size,
       'USE_BN': True,
       'USE_RESBLOCK': False,
       'LATENT_DIM': 128,
       'GRADIENT_PENALTY_LAMBDA': 10.0,
       'OUTPUT_DIR': 'output',
       'NET_DIM': 64,
       'TRAIN_ITERS': 20000,
       'DISC_LAMBDA': 0.0,
       'TV_LAMBDA': 0.0,
       'ATTRIBUTE': None,
       'TEST_BATCH_SIZE': 20,
       'NUM_GPUS': 1,
       'INPUT_TRANSFORM_TYPE': 0,
       'ENCODER_LR': 0.0002,
       'GENERATOR_LR': 0.0001,
       'DISCRIMINATOR_LR': 0.0004,
       'DISCRIMINATOR_REC_LR': 0.0004,
       'USE_ENCODER_INIT': True,
       'ENCODER_LOSS_TYPE': 'margin',
       'REC_LOSS_SCALE': 100.0,
       'REC_DISC_LOSS_SCALE': 1.0,
       'LATENT_REG_LOSS_SCALE': 0.5,
       'REC_MARGIN': 0.02,
       'ENC_DISC_TRAIN_ITER': 0,
       'ENC_TRAIN_ITER': 1,
       'DISC_TRAIN_ITER': 1,
       'GENERATOR_INIT_PATH': 'defence_gan/output/gans/mnist',
       'ENCODER_INIT_PATH': 'none',
       'ENC_DISC_LR': 1e-05,
       'NO_TRAINING_IMAGES': True,
       'GEN_SAMPLES_DISC_LOSS_SCALE': 1.0,
       'LATENTS_TO_Z_LOSS_SCALE': 1.0,
       'REC_CYCLED_LOSS_SCALE': 100.0,
       'GEN_SAMPLES_FAKING_LOSS_SCALE': 1.0,
       'DATASET_NAME': 'mnist',
       'ARCH_TYPE': 'mnist',
       'REC_ITERS': 200,
       'REC_LR': 0.01,
       'REC_RR': 1,
       'IMAGE_DIM': [28, 28, 1],
       'INPUR_TRANSFORM_TYPE': 1,
       'BPDA_ENCODER_CP_PATH': 'defence_gan/output/gans_inv_notrain/mnist',
       'BPDA_GENERATOR_INIT_PATH': 'defence_gan/output/gans/mnist',
       'cfg_path': 'experiments/cfgs/gans_inv_notrain/mnist.yml'
       }

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


def main(cfg, *args):
    FLAGS = tf.app.flags.FLAGS
    test_mode = not (FLAGS.is_train or FLAGS.train_encoder)
    # gan = DefenseGANv2(cfg=cfg, test_mode=test_mode)
    gan = DefenseGANv2(
        get_generator_fn(cfg['DATASET_NAME'], cfg['USE_RESBLOCK']), cfg=cfg,
        test_mode=test_mode,
    )

    if FLAGS.is_train:
        gan.train()

    if FLAGS.save_recs:
        ret_all = reconstruct_dataset(gan_model=gan, ckpt_path=FLAGS.init_path, max_num=FLAGS.max_num)
        save_mse(reconstruction_dict=ret_all, gan_model=gan)
        
    if FLAGS.test_generator:
        compute_inception_score(gan_model=gan, ckpt_path=FLAGS.init_path)

    if FLAGS.save_ds:
        save_ds(gan_model=gan)

    gan.close_session()


if __name__ == '__main__':
    args = parse_args()

    # Note: The load_config() call will convert all the parameters that are defined in
    # experiments/config files into FLAGS.param_name and can be passed in from command line.
    # arguments : python train.py --cfg <config_path> --<param_name> <param_value>
    cfg = load_config(args.cfg)
    flags = tf.app.flags

    flags.DEFINE_boolean("is_train", False,
                         "True for training, False for testing. [False]")
    flags.DEFINE_boolean("save_recs", False,
                         "True for saving reconstructions. [False]")
    flags.DEFINE_boolean("debug", False,
                         "True for debug. [False]")
    flags.DEFINE_boolean("test_generator", False,
                         "True for generator samples. [False]")
    flags.DEFINE_boolean("test_decoder", False,
                         "True for decoder samples. [False]")
    flags.DEFINE_boolean("test_again", False,
                         "True for not using cache. [False]")
    flags.DEFINE_boolean("test_batch", False,
                         "True for visualizing the batches and labels. [False]")
    flags.DEFINE_boolean("save_ds", False,
                         "True for saving the dataset in a pickle file. ["
                         "False]")
    flags.DEFINE_boolean("tensorboard_log", True, "True for saving "
                                                  "tensorboard logs. [True]")
    flags.DEFINE_boolean("train_encoder", False,
                         "Add an encoder to a pretrained model. ["
                         "False]")
    flags.DEFINE_boolean("test_encoder", False, "Test encoder. [False]")
    flags.DEFINE_boolean("init_with_enc", False,
                         "Initializes the z with an encoder, must run "
                         "--train_encoder first. [False]")
    flags.DEFINE_integer("max_num", -1,
                         "True for saving the dataset in a pickle file ["
                         "False]")
    flags.DEFINE_string("init_path", None, "Checkpoint path. [None]")

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)
