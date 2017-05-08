from utils import *
from network.download_file import download_file
import os


class Caffe(object):
    default_caffe_home = "caffe"
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
        execute(command)
        print ("Completed")

    def download_trained_model(self):
        trained_model_path = dir("trained_models/bvlc_reference_caffenet.caffemodel")
        if file_already_exists(trained_model_path):
            return
        print "Downloading trained model"
        download_file(self.trained_url, os.path.dirname(trained_model_path))
        save_checksum(trained_model_path)

    def train(self, solver, log):
        self.download_trained_model()
        trained_model_file = dir("trained_models/bvlc_reference_caffenet.caffemodel")
        solver = os.path.abspath(solver)
        log = os.path.abspath(log)
        caffe_bin = self.caffe_home() + "/build/tools/caffe"
        command = ["/usr/bin/nohup", caffe_bin, "train", "--solver=" + solver, "--weights", trained_model_file, "2>&1 >", log, "&"]
        command = ' '.join(command)
        execute_train_command(command)
        execute("tail -f " + log)
