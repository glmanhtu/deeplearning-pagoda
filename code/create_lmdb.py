'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''

import glob
import random

import cv2
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

from utils.percent_visualize import print_progress
from utils.utils import *


class CreateLmdb(object):

    def create_lmdb(self, train_path, train_lmdb_path, validation_lmdb_path, keyword):
        execute('rm -rf  ' + validation_lmdb_path)
        execute('rm -rf  ' + train_lmdb_path)
        train_data = [img for img in glob.glob(train_path + "/*jpg")]

        print 'Creating train_lmdb'

        # Shuffle train_data
        random.shuffle(train_data)

        in_db = lmdb.open(train_lmdb_path, map_size=int(1e12))
        total_elements = len(train_data)

        with in_db.begin(write=True) as in_txn:
            for in_idx, img_path in enumerate(train_data):
                if in_idx % 6 == 0:
                    continue
                self.save_lmdb(in_txn, in_idx, img_path, keyword)
                print_progress(in_idx, total_elements, "Progress:", "Complete", 2, 50)
        in_db.close()

        print '\nCreating validation_lmdb'

        in_db = lmdb.open(validation_lmdb_path, map_size=int(1e12))
        with in_db.begin(write=True) as in_txn:
            for in_idx, img_path in enumerate(train_data):
                if in_idx % 6 != 0:
                    continue
                self.save_lmdb(in_txn, in_idx, img_path, keyword)
                print_progress(in_idx, total_elements, "Progress:", "Complete", 2, 50)
        in_db.close()

        print '\nFinished processing all images'

    def save_lmdb(self, in_txn, in_idx, img_path, keyword):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=Constant.IMAGE_WIDTH, img_height=Constant.IMAGE_HEIGHT)
        if keyword in img_path:
            label = 0
        else:
            label = 1
        datum = self.make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())

    def make_datum(self, img, label):
        # image is numpy.ndarray format. BGR instead of RGB
        return caffe_pb2.Datum(
            channels=3,
            width=Constant.IMAGE_WIDTH,
            height=Constant.IMAGE_HEIGHT,
            label=label,
            data=np.rollaxis(img, 2).tostring())
