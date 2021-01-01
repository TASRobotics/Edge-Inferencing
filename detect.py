# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
import re
import tflite_runtime.interpreter as tflite
from flask import Flask, render_template, Response
from easydict import EasyDict as edict
from networktables import NetworkTables
import time

import logging


#Must enable logging to send data through networktables
logging.basicConfig(level=logging.DEBUG)

#Creates a flask app to run our server through
app = Flask(__name__)


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag.  gen_frames() is the function called for video feed.
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#Simply the location of our landing page when connecting to the Dev Board via HTTP, as well as the name of our HTML file.
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')




#Currently unecessary for a simple regression analysis
# These can be used again in the future for bounding boxes on object detection or even for classification algorithms.  
# In those cases modify the following four functions as necessary.
Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    return cv2_im



def gen_frames():  # generate frame by frame from camera
    #Current regression model is unstable, so we take running averages of sin, cos, and angle to stabilize the values
    moving_window = 10
    runsin = np.zeros(moving_window)
    runcos = np.zeros(moving_window)
    runtheta = np.zeros(moving_window)
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame
    
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
    
        common.set_input(interpreter, pil_im)
        interpreter.invoke()
        #Uncomment the following if you need to use bounding boxes or classification schemes and to use it for overlay
        
        #objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
        #cv2_im = append_objs_to_img(cv2_im, objs, labels)
        sincos = common.output_tensor(interpreter,0)
        runsin = np.roll(runsin,1)
        runsin[0]=sincos[0]
        runcos = np.roll(runcos,1)
        runcos[0]=sincos[1]
        runtheta = np.roll(runtheta,1)
        runtheta[0]=180/np.pi*np.arctan2(sincos[0],sincos[1])
        cv2_im = cv2.putText(cv2_im, 'angle: {}'.format(np.average(runtheta)),
                             (0,480-90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2_im = cv2.putText(cv2_im, 'sin: {}'.format(np.average(runsin)),
                             (0,480-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2_im = cv2.putText(cv2_im, 'cos: {}'.format(np.average(runcos)),
                             (0,480-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        sd.putNumber("WOF Angle", 180/np.pi*np.arctan2(np.average(runsin),np.average(runcos)))
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   

#Since the Dev board has two USB ports and the OS handles the numbering of them, we can't be sure of the camera identifier.
#These ids can either be 2 or 3 depending on order plugged in.  
#If using more than one camera at a time, this will have to be adjusted to properly identify the interpreter camera
            
def open_available_stream():
    print('Initializing Camera')
    for camera_id in range(2,4):
        cap = cv2.VideoCapture(camera_id)
        if cap is None or not cap.isOpened():
            print('Warning: unable to open video source: ', camera_id)
        else:
            print('Opening video source: ', camera_id)
            return cap
            
            
# Setting directories and filenames for models and labels
default_model_dir = './'
default_model = 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
default_labels = 'labels.txt'

#Most of these arguments are no longer necessary due to using openCV.
args = edict({'model':os.path.join(default_model_dir,default_model),
              'loop':True,
              'bitrate':1000000,
              'source':'/dev/video2:YUY2:640x480:30/1',
              'window':10,
              'top_k':3,
              'threshold':0.1,
              'print':False,
              'camera_idx':2,
              'labels':os.path.join(default_model_dir, default_labels)})


#Start up camera
cap = open_available_stream()

#Load up model, labels, and interpreter
print('Loading {} with {} labels.'.format(args.model, args.labels))
interpreter = common.make_interpreter(args.model)
interpreter.allocate_tensors()
labels = load_labels(args.labels)            

#Start networktables
NetworkTables.initialize()
sd = NetworkTables.getTable("SmartDashboard2")           
            
def main():
    
    #Run our flask app
    app.run(host='0.0.0.0',port=5000,use_reloader=False)
    
    #Cleaning up
    cap.release()
    cv2.destroyAllWindows()
    
    




if __name__ == '__main__':
    main()
