#!flask/bin/python
import random
import string

from flask import Flask, request
from utils.make_predictions import *

set_workspace("data/pagoda")
mean_proto = dir("data/mean.binaryproto")
caffe_deploy = dir("caffe_model/caffenet_deploy.prototxt")

render_template("template/caffenet_deploy.template", caffe_deploy)

mean_data = read_mean_data(mean_proto)
net = read_model_and_weight(caffe_deploy, dir("caffe_model/snapshot_iter_15000.caffemodel"))
transformer = image_transformers(net, mean_data)

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/predict', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            file_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            file_name = file_name + file.filename.rsplit('.', 1)[1]
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            file.save(file_path)
            predict = single_making_prediction(file_path, transformer, net)
            return predict

if __name__ == '__main__':
    app.run(debug=True)
