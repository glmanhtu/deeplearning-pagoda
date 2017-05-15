from network.download_google_drive import DownloadGoogleDrive
from network.google_file import GoogleFile
from utils.zip_utils import unzip_with_progress
from utils.create_lmdb import CreateLmdb
from utils.pycaffe import Caffe
from utils.make_predictions import *

google_download = DownloadGoogleDrive()

set_workspace("data/cat_dog")

train_zip = GoogleFile('0BzL8pCLanAIASFNnLUNEZFZHcmM', 'train.zip', dir('data/train.zip'))

print "\n\n------------------------PREPARE PHRASE----------------------------\n\n"

print "Starting download train file"
google_download.download_file_from_google_drive(train_zip)
print "Finish"

print "Extracting train zip file"
unzip_with_progress(train_zip.file_path, dir("data"))
print "Finish"


train_lmdb = dir("data/train_lmdb")
validation_lmdb = dir("data/validation_lmdb")

lmdb = CreateLmdb()
lmdb.create_lmdb(dir("data/train"), train_lmdb, validation_lmdb, "cat")

mean_proto = dir("data/mean.binaryproto")

caffe = Caffe()
caffe.compute_image_mean("lmdb", train_lmdb, mean_proto)
caffe.compute_image_mean("lmdb", validation_lmdb, mean_proto)

caffe_train_model = dir("caffe_model/caffenet_train.prototxt")
caffe_solver = dir("caffe_model/caffenet_solver.prototxt")

solver_mode = "GPU"
if "SOLVER_CPU" in os.environ:
    solver_mode = "CPU"

render_template("template/caffenet_train.template", caffe_train_model, mean_file=mean_proto,
                train_lmdb=train_lmdb, validation_lmdb=validation_lmdb, num_output=2)
render_template("template/caffenet_solver.template", caffe_solver, caffe_train_model=caffe_train_model,
                snapshot_prefix=dir("caffe_model/snapshot"), solver_mode=solver_mode, max_iterator=Constant.MAX_ITERATOR)

caffe_log = dir("caffe_model/caffe_train.log")

print "\n\n------------------------TRAINING PHRASE-----------------------------\n\n"

print "Starting to train"
caffe.train(caffe_solver, caffe_log)
print "Train completed"
