from network.download_google_drive import DownloadGoogleDrive
from network.google_file import GoogleFile
from utils.zip_utils import unzip_with_progress
from utils.make_predictions import *

google_download = DownloadGoogleDrive()

set_workspace("data/cat_dog")

test_zip = GoogleFile('0BzL8pCLanAIAZTlvcEs3U082U00', 'test1.zip', dir('data/test1.zip'))

print "\n\n------------------------PREPARE PHRASE----------------------------\n\n"

print "Starting download test file"
google_download.download_file_from_google_drive(test_zip)
print "Finish"

print "Extracting test zip file"
unzip_with_progress(test_zip.file_path, dir("data"))
print "Finish"

print "\n\n------------------------TESTING PHRASE-----------------------------\n\n"

caffe_deploy = dir("caffe_model/caffenet_deploy.prototxt")

render_template("template/caffenet_deploy.template", caffe_deploy)

mean_proto = dir("data/mean.binaryproto")

mean_data = read_mean_data(mean_proto)
net = read_model_and_weight(caffe_deploy, dir("caffe_model/snapshot_iter_10000.caffemodel"))
transformer = image_transformers(net, mean_data)
prediction = making_predictions(dir("data/test1"), transformer, net)

export_to_csv(prediction, dir("result/test_result.csv"))

print "\n\n-------------------------FINISH------------------------------------\n\n"
