from network.download_google_drive import DownloadGoogleDrive
from network.google_file import GoogleFile
from utils.zip_utils import unzip_with_progress
from code.create_lmdb import CreateLmdb
from utils.caffe import Caffe
from code.make_predictions import *

google_download = DownloadGoogleDrive()

set_workspace("data/cat_dog")

train_zip = GoogleFile('0BxsB7D9gLcdOQkdoQXRUMDdUUnM', 'train.zip', dir('data/train.zip'))
# test_zip = GoogleFile('0BzL8pCLanAIAZTlvcEs3U082U00', 'test1.zip', dir('data/test1.zip'))

print "\n\n------------------------PREPARE PHRASE----------------------------\n\n"

# print "Starting download test file"
# google_download.download_file_from_google_drive(test_zip)
# print "Finish"
#
# print "Extracting test zip file"
# unzip_with_progress(test_zip.file_path, dir("data"))
# print "Finish"

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
                train_lmdb=train_lmdb, validation_lmdb=validation_lmdb)
render_template("template/caffenet_solver.template", caffe_solver, caffe_train_model=caffe_train_model,
                snapshot_prefix=dir("caffe_model/snapshot"), solver_mode=solver_mode, max_iterator=Constant.MAX_ITERATOR)

caffe_log = dir("caffe_model/caffe_train.log")

print "\n\n------------------------TRAINING PHRASE-----------------------------\n\n"

print "Starting to train"
caffe.train(caffe_solver, caffe_log)
print "Train completed"

# print "\n\n------------------------TESTING PHRASE-----------------------------\n\n"
#
# caffe_deploy = dir("caffe_model/caffenet_deploy.prototxt")
#
# render_template("template/caffenet_deploy.template", caffe_deploy)
#
# mean_data = read_mean_data(mean_proto)
# net = read_model_and_weight(caffe_deploy, dir("caffe_model/caffe_model/snapshot_10000.caffemodel"))
# transformer = image_transformers(net, mean_data)
# prediction = making_predictions(dir("data/test1"), transformer, net)
#
# export_to_csv(prediction, dir("result/test_result.csv"))
#
# print "\n\n-------------------------FINISH------------------------------------\n\n"
