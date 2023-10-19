import requests
import time
import json
import uuid
import numpy as np
import cv2

class QueueController:
    def __init__(self, queue):
        self.queue_model = queue

    def process_get_image(self, uuid, cam_id):
        return self.queue_model.get_image(uuid, cam_id)
    
    def process_dequeue(self, inference_mode=True, variant="", suffix_no="", frame_no=""):
        return self.queue_model.dequeue(inference_mode, variant, suffix_no, frame_no)
    
    def process_get_image_length(self):
        return self.queue_model.get_image_length()
    
    def process_get_image_uuid(self):
        return self.queue_model.get_image_uuid()
    
    # Integrate capturing image and store it
    def generate(self):
        start = time.time()

        url_data = 'http://127.0.0.1:2123/capture_and_inference_frame'
        url_count_cam = 'http://127.0.0.1:2123/cameras_count'

        images_all, results_all = [], {}
        
        try:
            response_ccam = requests.get(url_count_cam)
            if response_ccam.status_code==200:
                count_cam = int(response_ccam.json())
        except Exception as e:
            print(e)

        partition = 100 / count_cam
        progress = 0

        try:
            response_data = requests.get(url_data)
            if response_data.status_code==200:
                data = response_data.json()
                images_all, results_all = np.array(data['images_all']), data['results_all']
        except Exception as e:
            print(e)

        for _ in range(count_cam):
            progress += partition
            progress_data = {
                'progress': progress,
                'isFinished' : False
            }
            yield 'data: {0}\n\n'.format(json.dumps(progress_data))

        uuid_image = uuid.uuid1()
        uuid_image = str(self.queue_model.get_image_length()) + str(uuid_image)

        self.queue_model.queue_images(uuid_image, images_all)

        print("All capture processes done")
        print(f"Time taken: {time.time() - start}")

        progress = 0
        progress_data = {
            'progress': progress,
            'isFinished' : True,
            'timeTaken': round(time.time() - start, 2)
        }

        yield 'data: {0}\n\n'.format(json.dumps(progress_data))

        self.queue_model.update_results(uuid_image, results_all)

    # Delete image from queue with index of idx
    def delete_image(self, idx):
        idx = int(idx)
        self.queue_model.delete(idx, True)
        return {
            "queue_length": self.queue_model.get_image_length(),
            "image_uuid" : self.queue_model.get_image_uuid(),
            "result" : self.queue_model.get_results()
        }
    
    # Get image from queue and pass it to FE
    def generate_image(self, image):
        _, jpeg = cv2.imencode(".jpg",image)
        image_ret = jpeg.tobytes()

        yield b'--frame\r\n'
        yield b'Content-Type: image/jpeg\r\n\r\n'
        yield image_ret
        yield b'\r\n'
                

