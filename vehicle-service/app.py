import json
import requests

from flask import Flask, Response, abort
from flask_cors import CORS

from src.models.queue import Queue
from src.models.specification import Specification
from src.controller.queue_controller import QueueController
from src.controller.specification_controller import SpecificationController

with open('vehicle_config.json', 'r') as file:
    config = json.load(file)

app = Flask(__name__)
CORS(app)

@app.get('/capture')
def capture():
    try:
        return Response(qc.generate(), mimetype='text/event-stream')
    except Exception as e:
        abort(e)

@app.get('/dequeue/<qr>')
def dequeue(qr):
    sc.process_set_specification(qr)

    result, uuid_image = qc.process_dequeue(True, 
                                            sc.specification_model.variant,
                                            sc.specification_model.suffix_no,
                                            sc.specification_model.frame_no,
                                            )
    
    sc.judgements(result, config['save_results'])

    return {
        'result' : result,
        'uuid_image' : uuid_image
    }

@app.post('/delete/<idx>')
def delete(idx):
    return qc.delete_image(idx)

@app.get('/image_list/<uuid>/<cam_id>')
def get_image_list(uuid, cam_id):
    cam_id = int(cam_id)
    image = qc.process_get_image(uuid, cam_id)
    return Response(qc.generate_image(image), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.get("/sync_queue")
def sync_queue():
    ci = []
    url_ci = 'http://127.0.0.1:2123/camera_indexes'

    try:
        resp = requests.get(url_ci)
        if resp.status_code==200:
            ci = resp.json()
    except Exception as e:
        print(e)

    return {
        "camera_list": ci,
        "queue_length": qc.process_get_image_length(),
        "image_uuid" : qc.process_get_image_uuid(),
        "vehicle_info" : {
            "qr" : sc.specification_model.qr,
            "variant" : sc.specification_model.variant,
            "suffix_no" : sc.specification_model.suffix_no,
            "frame_no" : sc.specification_model.frame_no
        }
    }


if __name__=='__main__':
    qc = QueueController(Queue())
    sc = SpecificationController(Specification(config['vehicle_name'],config['save_results']))
    app.run(host='0.0.0.0', port=2124,debug=False)