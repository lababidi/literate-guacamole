import os
import random

# from six.moves import cPickle as pickle
import pickle
from glob import glob

from tensorflow.python.platform import gfile

import tf_utils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
ANNOTATIONS = 'masks'  # "annotations"
IMAGES = "images"

def read_dataset(data_dir, download=False):
    pickle_filename = "MITSceneParsing.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        if download:
            utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
            scene_parsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
            data_dir = os.path.join(data_dir, scene_parsing_folder)
        result = create_image_lists(data_dir)
        print("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        training_records = result['training']
        validation_records = result['validation']
        del result

    return training_records, validation_records


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    # directories = ['training', 'validation']
    image_list = {'training': [], 'validation': []}

    for directory in image_list:
        file_list = glob(os.path.join(image_dir, directory, IMAGES, '*.jpg'))

        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("/")[-1])[0]
                annotation_file = os.path.join(image_dir, directory, ANNOTATIONS, filename + '.png')
                if os.path.exists(annotation_file):
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        print('No. of %s files: %d' % (directory, (len(image_list[directory]))))

    return image_list
