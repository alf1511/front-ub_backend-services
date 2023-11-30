import os
import cv2
import sys
import uuid
import time
import json
import threading

from flask import Flask, Response, jsonify, abort
from flask_cors import CORS
from flask_migrate import Migrate
from dotenv import load_dotenv

from ultralytics import YOLO
from utils import initialize_camera, inference_frame, get_newest_files

from specification import Specification
from queue_ import Queue
from vehicle import db

app = Flask(__name__)
CORS(app)

app.save_results = True
app.ai_model = {}
app.front_data = {}

app.is_inferencing = False

load_dotenv()

try:
    username = os.environ.get('POSTGRES_USER')
    password = os.environ.get('POSTGRES_PASSWORD')
    host = os.environ.get('POSTGRES_HOST')
    port = os.environ.get('POSTGRES_PORT')
    db_name = os.environ.get('POSTGRES_DB')
    
    app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{username}:{password}@{host}:{port}/{db_name}"

    db.init_app(app)
    db.session.expire_on_commit = False
    migrate = Migrate(app, db)
except Exception as e:
    app.save_results = False
    print("Failed to connect to database")

@app.get("/capture")
def capture():
    def capture_frame(camera_id):
        app.camera[camera_id].get_frame()
        if app.program_config["release_cam"]:
            app.camera[camera_id].release_camera()

    def generate():
        # initiate progress
        start = time.time()
        partition = 100 / len(app.camera)
        progress = 0

        # initiate images list
        images_all = []

        # Capture frame with multithreads if multithreads is enabled
        if app.program_config["multithreads"]['status']:
            threads = []
            max_threads = app.program_config["multithreads"]["max_threads"]

            for key in app.camera:
                thread = threading.Thread(target=capture_frame, args=(key,))
                threads.append(thread)

            for i in range(0, len(threads), max_threads):
                thread_group = threads[i:i + max_threads]
                for thread in thread_group:
                    thread.start()
                for thread in thread_group:
                    thread.join()

            if app.program_config["release_cam"]:
                for key in app.camera:
                    app.camera[key].release_camera()

            for key in app.camera:
                progress += partition
                progress_data = {
                    'progress': progress,
                    'isFinished': False
                }
                images_all.append(app.camera[key].original_frame)
                yield 'data: {0}\n\n'.format(json.dumps(progress_data))

        # Capture frame without multithreads if multithreads is disabled
        else:
            for key in app.camera:
                capture_frame(key)
                progress += partition
                progress_data = {
                    'progress': progress,
                    'isFinished': False,
                }
                images_all.append(app.camera[key].original_frame)
                yield 'data: {0}\n\n'.format(json.dumps(progress_data))

        uuid_image = uuid.uuid1()
        uuid_image = str(app.queue.get_image_length()) + str(uuid_image)

        app.queue.queue_images(uuid_image, images_all)

        print("All capture processes done")
        print(f"Time taken: {time.time() - start}")

        for key in app.camera:
            app.camera[key].reset()

        progress = 0
        progress_data = {
            'progress': progress,
            'isFinished': True,
            'timeTaken': round(time.time() - start, 2)
        }

        yield 'data: {0}\n\n'.format(json.dumps(progress_data))
            
    try:
        return Response(generate(), mimetype='text/event-stream')
    except Exception as e:
        abort(e)

@app.post('/delete/<idx>')
def delete(idx):
    idx = int(idx)
    app.queue.delete(idx)
    return {
        "queue_length": app.queue.get_image_length(),
        "image_uuid" : app.queue.get_image_uuid(),
        "result" : app.queue.get_results()
    }

@app.get("/image_list/<uuid>/<cam_id>")
def get_image_list(uuid, cam_id):
    cam_id = int(cam_id)
    def generate_image(image):
        _, jpeg = cv2.imencode(".jpg",image)
        image_ret = jpeg.tobytes()

        yield b'--frame\r\n'
        yield b'Content-Type: image/jpeg\r\n\r\n'
        yield image_ret
        yield b'\r\n'
    image = app.queue.get_image(uuid, cam_id)
    return Response(generate_image(image), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.get("/dequeue/<qr>")
def dequeue(qr):
    app.specification.set_specification(qr)
    app.front_data = app.specification.spec

    if app.inference_mode:
        app.is_inferencing = True
        top_uuid = app.queue.get_image_uuid()[0]
        vm = app.specification.variant

        inf_res = inference_frame(top_uuid, 
                                app.ai_model[vm],
                                app.model_config['conf'],
                                app.model_config['img_size'],
                                )
        app.queue.update_results(top_uuid, inf_res)

    result, uuid_image = app.queue.dequeue(app.inference_mode,
                                           app.specification.variant, 
                                           app.specification.suffix_no, 
                                           app.specification.frame_no)

    if result:
        app.specification.update_specification(result)

    if app.inference_mode:
        for k, v in result.items():
            part = k.split('-')[0].strip()
            if part in app.specification.spec:
                app.specification.spec[part][3].append('{:.3f}%'.format(v))
        
        if app.save_results:
            app.specification.save_vehicle()
            app.specification.save_specification()
        
        app.is_inferencing = False
    
    if 'ENGINE UNDER COVER' in app.specification.spec:
        if app.specification.spec['ENGINE UNDER COVER'][1]==[]:
            app.specification.spec['ENGINE UNDER COVER'][1].append('WITHOUT') 
            app.specification.spec['ENGINE UNDER COVER'][2] = 'OK' 
    
    print(f'new : {app.specification.spec}')

    app.front_data = app.specification.spec
    print(app.front_data)

    return {
        "result" : result,
        "uuid_image" : uuid_image
    }

@app.get("/sync_queue")
def sync_queue():
    return {
        "camera_list": app.camera_indexes,
        "queue_length": app.queue.get_image_length(),
        "image_uuid" : app.queue.get_image_uuid(),
        "vehicle_info" : {
            "qr" : app.specification.qr,
            "variant" : app.specification.variant,
            "suffix_no" : app.specification.suffix_no,
            "frame_no" : app.specification.frame_no
        }
    }

@app.get('/show_inf_images')
def show_inf_images():
    if os.path.exists(f"./INFERENCED_IMAGES"):
        data = get_newest_files('./INFERENCED_IMAGES', len(app.camera_indexes))
        print(data)
        return {
            'data': data
        }
    
    return {'resp': 'No inferenced image yet.'}

@app.get('/get_spec_front')
def get_spec_front():
    return app.front_data

@app.get('/reset_front_data')
def reset_front_data():
    app.front_data = {}

@app.get('/is_inferencing')
def is_inferencing():
    return {"data": app.is_inferencing}


if __name__ == '__main__':
    try:
        with open('config.json') as json_file:
            data = json.load(json_file)
            app.cam_config = data["cameras"]
            app.model_config = data["object_detection_model"]
            app.vehicle_name = data["vehicle_name"]
            app.program_config = data["program"]
            app.inference_mode = data["program"]["inference_mode"]
            app.cv_backend = data["program"]["cv_backend"]

            if app.save_results:
                app.save_results = data["program"]['save_results']
            else:
                app.program_config['save_results'] = False

            print("Config is loaded...")

            print("Camera List:")
            print(json.dumps(app.cam_config, indent=4))
            print()
            
            print("Object Detection Model:")
            print(json.dumps(app.model_config, indent=4))
            print()

            print(f"Vehicle Name Dictionary:")
            print(json.dumps(app.vehicle_name, indent=4))
            print()

            print(f"Program Config:")
            print(json.dumps(app.program_config, indent=4))
            print()

        for k,v in data['object_detection_model']['model'].items():
            app.ai_model[k] = YOLO(v)

        # initialize camera
        app.camera, app.camera_indexes = initialize_camera(app.cam_config, app.program_config, app.cv_backend)

        if len(app.camera_indexes) == 0:
            print("No camera available")
            print("Exiting ...")
            sys.exit()

        # initialize specification
        app.specification = Specification(app.vehicle_name, app.save_results)

        # initialize queue
        app.queue = Queue()

        app.run(host='0.0.0.0', port=2124, debug=False)

    except Exception as e:
        print(e)
        print("Error reading config.json")
        print("Please check config.json file")
        print("Exiting ...")
        sys.exit()