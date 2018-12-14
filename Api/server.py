import json, argparse

import tensorflow as tf
import os

from ml_lib.load import load_graph
from ml_lib.ocr import process_img
from ml_lib.nn import predict
from ml_lib.map import make_output_map

from flask import Flask, request, render_template, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))

    return render_template('index.html', all_files = os.listdir('static/uploads'))

@app.route('/api/<filename>/analyze')
def analyze(filename):
    # process img
    data = process_img('./static/uploads/' + filename)
    return predict(data, persistent_sess, x, y, CNN_output_map)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('result.html', filename=filename)

@app.route('/uploads/<filename>/delete')
def remove_file(filename):
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect('/')
    
if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="frozen_graph_v1.pb", type=str, help="Frozen model file to import")
    parser.add_argument("--gpu_memory", default=.2, type=float, help="GPU memory per process")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)
        
    # We access the input and output nodes 
    x = graph.get_tensor_by_name('prefix/Reshape:0')
    y = graph.get_tensor_by_name('prefix/ArgMax:0')

    # init output map
    CNN_output_map = make_output_map()
        
    print('Starting Session, setting the GPU memory usage to %f' % args.gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=graph, config=sess_config)

    # run api

    print('Starting the API')
    app.run(port=5000, debug=True)