import sys
import time
import json

from flask import Flask, jsonify
from flask_cors import CORS

from ultralytics import YOLO

from src.controllers.camera_controller import CameraController
from src.models.camera import Camera

with open('camera_config.json', 'r') as file:
    config = json.load(file)

app = Flask(__name__)
CORS(app)

@app.get("/capture_and_inference_frame")
def capture_and_inference_frame(): 
    camera_controller.reset_images_results()
    camera_controller.capture_images(config['program']['multithreads']['max_threads'])
    results_all = camera_controller.get_results()
    images_all = [image.tolist() for image in camera_controller.get_images()]

    response_data = {
        "images_all": images_all,
        "results_all": results_all
    }

    return jsonify(response_data) 

@app.get('/cameras_count')
def cameras_count():
     return jsonify(len(app.cameras))

@app.get('/camera_indexes')
def camera_indexes():
    return app.camera_indexes
         
        
if __name__=='__main__':
    start = time.time()

    app.ai_model = YOLO(config['object_detection_model']['model'])

    app.cameras = {}
    app.camera_indexes = []
    
    conf = config['object_detection_model']['conf']
    img_size = config['object_detection_model']['img_size']
    cv_backend = config['program']['cv_backend']

    for index, cam in config['cameras'].items():
        camera = Camera(index, cam, app.ai_model, conf, img_size, cv_backend)

        if camera.isOpened():
            app.cameras[index] = camera
            app.camera_indexes.append(index)

        if config['program']['release_cam']:
            camera.release_camera()
    
    print(f'Available cameras : {app.cameras}')

    camera_controller = CameraController(app.cameras, True, config['program']['release_cam'])

    if len(app.cameras) == 0:
            print("No camera available")
            print("Exiting ...")
            sys.exit()

    print(time.time()-start)

    app.run(host='0.0.0.0', port=2123,debug=False)