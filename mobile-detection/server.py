from http.server import BaseHTTPRequestHandler
import socketserver
import cgi
import codecs
import os

import numpy as np

from app.utils import detect_image_http
import tensorflow as tf
import json
yolo = tf.keras.models.load_model('yolo2.h5')
YOLO_INPUT_SIZE = 640
TRAIN_CLASSES = "../model_data/names.txt"


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

    # return string
    return str1

class Server(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'
        try:
            split_path = os.path.splitext(self.path)
            request_extension = split_path[1]
            if request_extension != ".py":
                f = open(self.path[1:]).read()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(bytes(f, 'utf-8'))
            else:
                f = "File not found"
                self.send_error(404,f)
        except:
            f = "File not found"
            self.send_error(404,f)
    def do_POST(self):
        print("command: " + self.command + "\npath: " + self.path)
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                     })
        filename = form['file'].filename
        data = form['file'].file.read()
        split_tup = os.path.splitext(filename)
        image_path = 'input' + split_tup[1]
        output_path = 'output' + split_tup[1]
        with open(image_path, "wb") as fh:
            fh.write(data)
        print('image_path', image_path)
        print('output_path', output_path)
        print('split_tup', split_tup)
        print('filename', filename)
        res = detect_image_http(
            yolo,
            image_path,
            output_path,
            input_size=YOLO_INPUT_SIZE,
            show=True,
            CLASSES=TRAIN_CLASSES,
            rectangle_colors=(255, 0, 0),
            score_threshold=0.4
        )
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, PUT, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()

        self.wfile.write(bytes(json.dumps({"res": np.array(res).tolist()}), encoding='utf8'))
    def do_OPTIONS(self):
        self.do_POST()

