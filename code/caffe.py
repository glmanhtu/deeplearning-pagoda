from subprocess import call
from utils.utils import *
from network.download_file import download_file
import os


class Caffe(object):
    default_caffe_home = "/home/ubuntu/caffe"
    trained_url = "http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel"

    def caffe_home(self):
        if "CAFFE_ROOT" in os.environ:
            return os.environ['CAFFE_ROOT']
        return self.default_caffe_home

    def compute_image_mean(self, backend, lmbd_path, binaryproto_path):
        print ("Computing image mean from %s" % lmbd_path)
        image_mean_bin = self.caffe_home() + "/build/tools/compute_image_mean"
        lmbd_path = os.path.abspath(lmbd_path)
        binaryproto_path = os.path.abspath(binaryproto_path)
        command = [image_mean_bin, "-backend=" + backend, lmbd_path, binaryproto_path]
        command = ' '.join(command)
        call(command, shell=True)
        print ("Completed")

    def download_trained_model(self):
        trained_model_path = self.caffe_home() + "/models/bvlc_reference_caffenet"
        if file_already_exists(trained_model_path + "/bvlc_reference_caffenet.caffemodel"):
            return
        print "Downloading trained model"
        download_file(self.trained_url, trained_model_path)
        save_checksum(trained_model_path + "/bvlc_reference_caffenet.caffemodel")

    def train(self, solver, log):
        trained_model_path = self.caffe_home() + "/models/bvlc_reference_caffenet"
        trained_model_file = trained_model_path + "/bvlc_reference_caffenet.caffemodel"
        solver = os.path.abspath(solver)
        log = os.path.abspath(log)
        caffe_bin = self.caffe_home() + "/build/tools/caffe"
        command = [caffe_bin, "train", "--solver=" + solver, "--weights", trained_model_file, "2>&1 | tee", log]
        command = ' '.join(command)
        execute_train_command(command)