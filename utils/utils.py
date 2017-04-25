import os
import hashlib
import subprocess
import sys
from mako.template import Template
import cv2
from constants import Constant
import numbers

constant = Constant()


def set_workspace(ws):
    constant.set_workspace(ws)
    if not os.path.isdir(ws):
        os.makedirs(ws)


def dir(path):
    return constant.get_workspace() + "/" + path


def file_already_exists(file_path):
    if os.path.isfile(file_path):
        checksum_file = file_path + ".checksum"
        if os.path.isfile(checksum_file):
            checksum_original = open(checksum_file, 'r').read()
            checksum = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
            if checksum_original == checksum:
                return True
            os.remove(checksum_file)
        os.remove(file_path)
    return False


def human_2_bytes(s):
    """
    >>> human2bytes('1M')
    1048576
    >>> human2bytes('1G')
    1073741824
    """
    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    letter = s[-1:].strip().upper()
    num = s[:-1]
    assert num.isdigit() and letter in symbols
    num = float(num)
    prefix = {symbols[0]:1}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    return int(num * prefix[letter])


def render_template(template_file, destination_file, **data):
    template = Template(filename=template_file)
    for parameter in data:
        if not isinstance(data[parameter], numbers.Number):
            if os.path.isdir(data[parameter]) or os.path.isfile(data[parameter]):
                data[parameter] = os.path.abspath(data[parameter])
    result = template.render(**data)
    with open(destination_file, "w") as text_file:
        text_file.write(result)


def save_checksum(file_path):
    checksum = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
    with open(file_path + ".checksum", "w") as text_file:
        text_file.write(checksum)


def execute(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Poll process for new output until finished
    while True:
        next_line = process.stdout.readline()
        if next_line == '' and process.poll() is not None:
            break
        sys.stdout.write(next_line)
        sys.stdout.flush()


def execute_train_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Poll process for new output until finished
    while True:
        next_line = process.stdout.readline()
        if next_line == '' and process.poll() is not None:
            break
        sys.stdout.write(next_line)
        sys.stdout.flush()


def transform_img(img, img_width, img_height):
    # Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img