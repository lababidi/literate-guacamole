from __future__ import print_function

import gdal
import datetime
import os
import random
import sys
import tarfile
import zipfile
from glob import glob
from os.path import join, splitext, exists

import numpy as np
import requests
import scipy.io
import tensorflow as tf
from scipy.misc import imread, imsave, imresize
from tensorflow.python.platform import gfile

import rasterio

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_string("train_dir", "", "path to dataset")
tf.flags.DEFINE_string("val_dir", "", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "True", "Debug mode: True/ False")
tf.flags.DEFINE_bool('resize', "True", "Resize Images: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
tf.flags.DEFINE_integer('channels', "8", "number of channels in image")
tf.flags.DEFINE_integer('size', "768", "image size in pixels to resize to")
tf.flags.DEFINE_string('ext', "tif", "image extension (Not masks)")

DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSES = 2
IMAGE_SIZE = 768
IMG_CHANNELS = FLAGS.channels

ANNOTATIONS = 'masks'  # "annotations"
IMAGES = "images"


def gen_file_path(file_url, dir_path):
    filename = file_url.split('/')[-1]
    file_path = os.path.join(dir_path, filename)
    return filename, file_path


def extract(dir_path, file_path, ziptar=None):
    if ziptar == "tar":
        tarfile.open(file_path, 'r:gz').extractall(dir_path)
    elif ziptar == "zip":
        with zipfile.ZipFile(file_path) as zf:
            zf.extractall(dir_path)


def download(path, url):
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        blocks, chunk_size = 0, 1024
        total_length = int(r.headers.get('content-length'))
        total_blocks = total_length / chunk_size
        for chunk in r.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                blocks += 1
                sys.stdout.write('\r>> Downloading {}'.format(100.0 * blocks / total_blocks))
                sys.stdout.flush()
        return os.stat(path).st_size


def ww(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bb(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def conv2d(x, w, bias):
    return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME"), bias)


def download_ade_date(data_dir):
    filename, file_path = gen_file_path(DATA_URL, data_dir)
    if not exists(file_path):
        os.makedirs(data_dir, exist_ok=True)
        file_size = download(file_path, DATA_URL)
        print('Successfully downloaded', filename, file_size, 'bytes.')
        extract(data_dir, file_path, ziptar='zip')

    scene_parsing_folder = splitext(DATA_URL.split("/")[-1])[0]
    return join(data_dir, scene_parsing_folder)


class Record:
    def __init__(self, image, mask):
        print(image, mask)
        self.image = image
        self.mask = mask


class BatchDataset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, directory, ext, resize, size, mask_ext='png', filename=None):
        """
        Initialize a generic file reader with batching for list of files
        :param mask_ext:
        :param ext:
        :param directory:
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        records = []
        if gfile.Exists(join(directory, filename)):
            with open(join(directory, filename)) as f:
                file_list = [join(directory, IMAGES, image) for image in f.read().split()]
        else:
            file_list = glob(join(directory, IMAGES, '*.{}'.format(ext)))

        if not file_list:
            print('No files found')
            raise FileNotFoundError(join(directory, IMAGES, '*.{}'.format(ext)))
        else:
            for f in file_list:
                filename = splitext(f.split("/")[-1])[0]
                mask_file = join(directory, ANNOTATIONS, filename + '.' + mask_ext)
                if exists(mask_file):
                    records.append(Record(f, mask_file))
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(records)
        self.records = records
        print('No. of %s files: %d' % (directory, (len(records))))

        self.resize = resize
        self.size = size
        print("Initializing Batch Dataset Reader...")
        # todo TIF reading and manipulation
        if ext == 'tif':
            self.images = np.array([self.transform_tif(record.image) for record in records])
        else:
            self.images = np.array([self.transform(imread(record.image)) for record in records])
        self.annotations = np.array(
            [np.expand_dims(self.transform_mask(imread(record.mask)), axis=3) for record in records])
        self.batch_offset = 0
        print(self.images.shape)
        print(self.annotations.shape)

    def transform(self, image, mask=False):
        if self.resize:
            image = np.array(imresize(image, [self.size, self.size], interp='nearest'))

        if not mask and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image] * IMG_CHANNELS)
        elif not mask and image.shape[-1] < IMG_CHANNELS:  # make sure images are of shape(h,w,3)
            image = np.dstack([image] * (1 + IMG_CHANNELS // image.shape[-1]))[:, :, :IMG_CHANNELS]

        if mask and image.shape[-1] > 1:
            # image = np.dot(image[..., :3], [0.299/128, 0.587/128, 0.114/128])
            image = (image.sum(axis=-1) > 0).astype(np.int)

        return np.array(image)

    def transform_mask(self, image):
        if self.resize:
            image = np.array(imresize(image, [self.size, self.size], interp='nearest'))

        if len(image.shape) > 2 and image.shape[-1] > 1:
            # image = np.dot(image[..., :3], [0.299/128, 0.587/128, 0.114/128])
            image = (image.sum(axis=-1) > 0).astype(np.int)
        image = (image > 0).astype(np.int)

        return np.array(image)

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes], [self.records[index] for index in indexes]

    def transform_tif(self, image):
        with rasterio.open(image) as f:
            i = np.array(f.read())

        # i = gdal.Open(image, gdal.GA_ReadOnly).ReadAsArray()
        if self.resize:
            new_image = []
            for layer in i:
                new_layer = np.array(imresize(layer, [self.size, self.size], interp='nearest'))
                new_image.append(new_layer)
            i = np.array(new_image).transpose([1, 2, 0])
        return i


def vgg_net(weights, image, mean):
    def extract_var(w, var_name, trainable=True):
        return tf.get_variable(name=var_name,
                               initializer=(tf.constant_initializer(w)),
                               shape=w.shape,
                               trainable=trainable)

    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current_layer = image  # - mean
    for i, name in enumerate(layers):
        kind = name[:4]
        if i == 0:
            rgb_kernels, bias = weights[i][0][0][0][0]
            # mat-convnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            print("kernel", rgb_kernels.shape)
            # reshape initial Convolution Kernel from 3 channels to IMG_CHANNELS through copying (r,g,b,r,g,b,r,g,...)
            # this is slightly clunky, but until we have a better VGG model for multiband this will do
            # rgb_kernels = np.concatenate(([rgb_kernels] * IMG_CHANNELS), axis=2)[:, :, :IMG_CHANNELS, :]
            if IMG_CHANNELS == 8:
                new_shape = list(rgb_kernels.shape)
                new_shape[2] = 8
                kernels = np.zeros(new_shape)
                kernels[:, :, 0, :] = rgb_kernels[:, :, 0, :]
                kernels[:, :, 1, :] = rgb_kernels[:, :, 0, :]
                kernels[:, :, 2, :] = rgb_kernels[:, :, 1, :]
                kernels[:, :, 3, :] = rgb_kernels[:, :, 1, :]
                kernels[:, :, 4, :] = rgb_kernels[:, :, 2, :]
                kernels[:, :, 5, :] = rgb_kernels[:, :, 2, :]
                kernels[:, :, 6, :] = rgb_kernels[:, :, 2, :]
                kernels[:, :, 7, :] = rgb_kernels[:, :, 2, :]
            else:
                kernels = rgb_kernels

            print("kernel", kernels.shape)
            kernels = extract_var(np.transpose(kernels, (1, 0, 2, 3)), var_name=name + "_w")
            bias = extract_var(bias.reshape(-1), var_name=name + "_b")
            current_layer = conv2d(current_layer, kernels, bias)
        elif kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # mat-convnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = extract_var(np.transpose(kernels, (1, 0, 2, 3)), var_name=name + "_w")
            bias = extract_var(bias.reshape(-1), var_name=name + "_b")
            current_layer = conv2d(current_layer, kernels, bias)
        elif kind == 'relu':
            current_layer = tf.nn.relu(current_layer, name=name)
            if FLAGS.debug:
                # utils.add_activation_summary(current)
                tf.summary.histogram(current_layer.op.name + "/activation", current_layer)
                tf.summary.scalar(current_layer.op.name + "/sparsity", tf.nn.zero_fraction(current_layer))
        elif kind == 'pool':
            current_layer = tf.nn.avg_pool(current_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        net[name] = current_layer

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")

    filename, file_path = gen_file_path(MODEL_URL, FLAGS.model_dir)
    if not exists(file_path):
        os.makedirs(FLAGS.model_dir, exist_ok=True)
        file_size = download(file_path, MODEL_URL)
        print('Successfully downloaded', filename, file_size, 'bytes.')
        extract(FLAGS.model_dir, file_path)

    if not exists(file_path):
        raise IOError("VGG Model not found!")
    model_data = scipy.io.loadmat(file_path)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    mean_pixel = np.concatenate([mean_pixel, mean_pixel])

    weights = np.squeeze(model_data['layers'])

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, image, mean_pixel)
        conv_final_layer = image_net["conv5_3"]

        pool5 = tf.nn.max_pool(conv_final_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        w6 = ww([7, 7, 512, 4096], name="W6")
        b6 = bb([4096], name="b6")
        conv6 = conv2d(pool5, w6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            # utils.add_activation_summary(relu6)
            tf.summary.histogram(relu6.op.name + "/activation", relu6)
            tf.summary.scalar(relu6.op.name + "/sparsity", tf.nn.zero_fraction(relu6))
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        w7 = ww([1, 1, 4096, 4096], name="W7")
        b7 = bb([4096], name="b7")
        conv7 = conv2d(relu_dropout6, w7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            # utils.add_activation_summary(relu7)
            tf.summary.histogram(relu7.op.name + "/activation", relu7)
            tf.summary.scalar(relu7.op.name + "/sparsity", tf.nn.zero_fraction(relu7))
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        w8 = ww([1, 1, 4096, NUM_OF_CLASSES], name="W8")
        b8 = bb([NUM_OF_CLASSES], name="b8")
        conv8 = conv2d(relu_dropout7, w8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        w_t1 = ww([4, 4, deconv_shape1[3].value, NUM_OF_CLASSES], name="W_t1")
        b_t1 = bb([deconv_shape1[3].value], name="b_t1")
        conv_t1 = tf.nn.bias_add(
            tf.nn.conv2d_transpose(conv8, w_t1, tf.shape(image_net["pool4"]),
                                   strides=[1, 2, 2, 1], padding="SAME"),
            b_t1)

        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        w_t2 = ww([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = bb([deconv_shape2[3].value], name="b_t2")
        conv_t2 = tf.nn.bias_add(
            tf.nn.conv2d_transpose(fuse_1, w_t2, tf.shape(image_net["pool3"]), strides=[1, 2, 2, 1], padding="SAME"),
            b_t2)
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSES])
        w_t3 = ww([16, 16, NUM_OF_CLASSES, deconv_shape2[3].value], name="W_t3")
        b_t3 = bb([NUM_OF_CLASSES], name="b_t3")
        conv_t3 = tf.nn.bias_add(
            tf.nn.conv2d_transpose(fuse_2, w_t3, deconv_shape3, strides=[1, 8, 8, 1], padding="SAME"), b_t3)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3, image_net


def train_optimization(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradient", grad)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    print(argv)

    print("Setting up dataset reader")
    if not FLAGS.val_dir or not FLAGS.train_dir:
        train_dir = val_dir = FLAGS.data_dir
        train_batch = "train.txt"
        val_batch = "val.txt"
    else:
        val_dir = FLAGS.val_dir
        train_dir = FLAGS.train_dir
        train_batch = val_batch = None

    validation_dataset = BatchDataset(val_dir, FLAGS.ext, FLAGS.resize, FLAGS.size, filename=val_batch)

    print("Setting up image reader...")

    keep_probability = tf.placeholder(tf.float32, name="keep_probability")
    image_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMG_CHANNELS],
                                       name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits, image_net = inference(image_placeholder, keep_probability)
    # tf.summary.image("input_image", image_placeholder, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3]), name="entropy")))
    tf.summary.scalar("entropy", loss)

    if FLAGS.debug:
        for var in tf.trainable_variables():
            # utils.add_to_regularization_and_summary(var)
            tf.summary.histogram(var.op.name, var)
            tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss, var_list=(tf.trainable_variables()))
    if FLAGS.debug:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + "/gradient", grad)
    train_optimizer = optimizer.apply_gradients(grads)

    summary_op = tf.summary.merge_all()

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        train_dataset = BatchDataset(train_dir, FLAGS.ext, True, IMAGE_SIZE, filename=train_batch)
        for itr in range(MAX_ITERATION):
            train_images, train_annotations = train_dataset.next_batch(FLAGS.batch_size)
            feed_dict = {image_placeholder: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_optimizer, feed_dict=feed_dict)

            if itr % 1 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
                tf.train.export_meta_graph('fcn.tff')

            if itr % 5 == 0:
                valid_images, valid_annotations = validation_dataset.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image_placeholder: valid_images,
                                                       annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                # saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        # with open('conv1.dat', 'w') as f:
        #     f.write(image_net['conv1_1'].eval(session=sess))
        sess.as_default()
        valid_images, valid_annotations, recs = validation_dataset.get_random_batch(FLAGS.batch_size)
        np.save('conv1_1w', (sess.run(tf.trainable_variables()[0])))
        np.save('conv1_1b', (sess.run(tf.trainable_variables()[1])))
        np.save('conv1_2w', (sess.run(tf.trainable_variables()[3])))
        np.save('conv1_2b', (sess.run(tf.trainable_variables()[4])))
        np.save('conv1.out',
                sess.run(image_net['conv1_1'], feed_dict={image_placeholder: valid_images}).astype(np.float16))
        time_now = datetime.datetime.now()
        pred = sess.run(pred_annotation, feed_dict={image_placeholder: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        print(datetime.datetime.now() - time_now)
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            # imsave(join(FLAGS.logs_dir, "inp_{}.png".format(itr)), valid_images[itr].astype(np.uint8))
            with open(join(FLAGS.data_dir, 'inp_{}.txt'.format(itr)), 'w') as f:
                f.write(recs[itr].image)
                f.write(recs[itr].mask)
            imsave(join(FLAGS.data_dir, "gt_{}.png".format(itr)), valid_annotations[itr].astype(np.uint8))
            imsave(join(FLAGS.data_dir, "pred_{}.png".format(itr)), pred[itr].astype(np.uint8))
            print("Saved image: %d" % itr)

        tf.train.write_graph(sess.graph_def, 'models/', 'graph.pb', as_text=False)
        tf.train.write_graph(sess.graph_def, 'models/', 'graph.pbtx')
        tf.train.export_meta_graph('fcn.tf', collection_list=['pred_annotation', 'loss', 'logits'])


if __name__ == "__main__":
    tf.app.run()
