import os
import scipy.misc
import numpy as np
import tensorflow as tf
from tflib.layers import *

def mnist_generator(z, is_training=True):
    net_dim = 64
    use_sn = False
    update_collection = None
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        output = linear(z, 4 * 4 * 4 * net_dim, sn=use_sn, name='linear')
        output = batch_norm(output, is_training=is_training, name='bn_linear')
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4, 4, 4 * net_dim])

        # deconv-bn-relu
        output = deconv2d(output, 2 * net_dim, 5, 2, sn=use_sn, name='deconv_0')
        output = batch_norm(output, is_training=is_training, name='bn_0')
        output = tf.nn.relu(output)

        output = output[:, :7, :7, :]

        output = deconv2d(output, net_dim, 5, 2, sn=use_sn, name='deconv_1')
        output = batch_norm(output, is_training=is_training, name='bn_1')
        output = tf.nn.relu(output)

        output = deconv2d(output, 1, 5, 2, sn=use_sn, name='deconv_2')
        output = tf.sigmoid(output)

        return output

GENERATOR_DICT = {'mnist': [mnist_generator, mnist_generator]}

class Dataset(object):
    """The abstract class for handling datasets.

    Attributes:
        name: Name of the dataset.
        data_dir: The directory where the dataset resides.
    """

    def __init__(self, name, data_dir='./data'):
        """The datasaet default constructor.

            Args:
                name: A string, name of the dataset.
                data_dir (optional): The path of the datasets on disk.
        """

        self.data_dir = os.path.join(data_dir, name)
        self.name = name
        self.images = None
        self.labels = None

    def __len__(self):
        """Gives the number of images in the dataset.

        Returns:
            Number of images in the dataset.
        """

        return len(self.images)

    def load(self, split):
        """ Abstract function specific to each dataset."""
        pass

class Mnist(Dataset):
    """Implements the Dataset class to handle MNIST.

    Attributes:
        y_dim: The dimension of label vectors (number of classes).
        split_data: A dictionary of
            {
                'train': Images of np.ndarray, Int array of labels, and int
                array of ids.
                'val': Images of np.ndarray, Int array of labels, and int
                array of ids.
                'test': Images of np.ndarray, Int array of labels, and int
                array of ids.
            }
    """

    def __init__(self):
        super(Mnist, self).__init__('mnist')
        self.y_dim = 10
        self.split_data = {}

    def load(self, split='train', lazy=True, randomize=True):
        """Implements the load function.

        Args:
            split: Dataset split, can be [train|dev|test], default: train.
            lazy: Not used for MNIST.

        Returns:
             Images of np.ndarray, Int array of labels, and int array of ids.

        Raises:
            ValueError: If split is not one of [train|val|test].
        """

        if split in self.split_data.keys():
            return self.split_data[split]

        data_dir = self.data_dir

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        train_images = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        train_labels = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        test_images = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        test_labels = loaded[8:].reshape((10000)).astype(np.float)

        train_labels = np.asarray(train_labels)
        test_labels = np.asarray(test_labels)
        if split == 'train':
            images = train_images[:50000]
            labels = train_labels[:50000]
        elif split == 'val':
            images = train_images[50000:60000]
            labels = train_labels[50000:60000]
        elif split == 'test':
            images = test_images
            labels = test_labels

        if randomize:
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
        images = np.reshape(images, [-1, 28, 28, 1])
        self.split_data[split] = [images, labels]
        self.images = images
        self.labels = labels

        return images, labels


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate

def create_generator(dataset_name, split, batch_size, randomize,
                     attribute=None):
    """Creates a batch generator for the dataset.

    Args:
        dataset_name: `str`. The name of the dataset.
        split: `str`. The split of data. It can be `train`, `val`, or `test`.
        batch_size: An integer. The batch size.
        randomize: `bool`. Whether to randomize the order of images before
            batching.
        attribute (optional): For cele

    Returns:
        image_batch: A Python generator for the images.
        label_batch: A Python generator for the labels.
    """
    flags = tf.app.flags.FLAGS

    if dataset_name.lower() == 'mnist':
        ds = Mnist()
    else:
        raise ValueError("Dataset {} is not supported.".format(dataset_name))

    ds.load(split=split, randomize=randomize)

    def get_gen():
        for i in range(0, len(ds) - batch_size, batch_size):
            image_batch, label_batch = ds.images[
                                       i:i + batch_size], \
                                       ds.labels[i:i + batch_size]
            yield image_batch, label_batch

    return get_gen

def get_generators(dataset_name, batch_size, randomize=True, attribute='gender'):
    """Creates batch generators for datasets.

    Args:
        dataset_name: A `string`. Name of the dataset.
        batch_size: An `integer`. The size of each batch.
        randomize: A `boolean`.
        attribute: A `string`. If the dataset name is `celeba`, this will
         indicate the attribute name that labels should be returned for.

    Returns:
        Training, validation, and test dataset generators which are the
            return values of `create_generator`.
    """
    splits = ['train', 'val', 'test']
    gens = []
    for i in range(3):
        if i > 0:
            randomize = False
        gens.append(
            create_generator(dataset_name, splits[i], batch_size, randomize,
                             attribute=attribute))

    return gens

def save_image_core(image, path):
    """Save an image as a png file"""
    if image.shape[0] == 3 or image.shape[0] == 1:
        image = image.transpose([1, 2, 0])
    image = ((image.squeeze() * 1.0 - image.min()) / (
        image.max() - image.min() + 1e-7)) * 255
    image = image.astype(np.uint8)
    scipy.misc.imsave(path, image)

    print('[#] saved image to: {}'.format(path))

@static_vars(image_counter=0)
def save_image(image, fname=None, dir_path='debug/images/'):
    if fname is None:
        fname = 'image_{}.png'.format(save_image.image_counter)
        save_image.image_counter = save_image.image_counter + 1
    make_dir(dir_path)
    fpath = os.path.join(dir_path, fname)
    save_image_core(image, fpath)

def get_generator_fn(dataset_name, use_resblock=False):
    if use_resblock:
        return GENERATOR_DICT[dataset_name][1]
    else:
        return GENERATOR_DICT[dataset_name][0]

def save_images_files(images, prefix='im', labels=None, output_dir=None,
                      postfix=''):
    if prefix is None and labels is None:
        prefix = '{}_image.png'
    else:
        prefix = prefix + '_{:03d}'
    if labels is not None:
        prefix = prefix + '_{:03d}'

    prefix = prefix + postfix + '.png'

    assert len(images.shape) == 4, 'images should be a 4D np array uint8'
    for i in range(images.shape[0]):
        image = images[i]
        if labels is None:
            save_image(image, fname=prefix.format(i), dir_path=output_dir)
        else:
            save_image(image, fname=prefix.format(i, int(labels[i])),
                       dir_path=output_dir)

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('[+] Created the directory: {}'.format(dir_path))


ensure_dir = make_dir