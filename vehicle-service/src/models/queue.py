import cv2
import os
import shutil
import pickle as pkl

from .utils.save_capture import saveImageCapture

class Queue:
    def __init__(self):
        self.image_uuid, self.results = self._init_queue()

    def _init_queue(self):
        image_uuid = []
        results = {}
        if os.path.exists(f"./TEMP_QUEUE"):
            if os.path.exists("./TEMP_QUEUE/image_uuid.pkl"):
                with open(f'./TEMP_QUEUE/image_uuid.pkl', 'rb') as file:
                    image_uuid = pkl.load(file)
                with open(f'./TEMP_QUEUE/results.pkl', 'rb') as file:
                    results = pkl.load(file)
            else:
                with open(f'./TEMP_QUEUE/image_uuid.pkl', 'wb') as file:
                    pkl.dump(image_uuid, file)
                with open(f'./TEMP_QUEUE/results.pkl', 'wb') as file:
                    pkl.dump(results, file)
        else:
            os.makedirs(f"TEMP_QUEUE")
            with open(f'./TEMP_QUEUE/image_uuid.pkl', 'wb') as file:
                    pkl.dump(image_uuid, file)
            with open(f'./TEMP_QUEUE/results.pkl', 'wb') as file:
                pkl.dump(results, file)
        return image_uuid, results
    
    def _sync_image_uuid(self):
        with open(f'./TEMP_QUEUE/image_uuid.pkl', 'wb') as file:
            pkl.dump(self.image_uuid, file)

    def _sync_results(self):
        with open(f'./TEMP_QUEUE/results.pkl', 'wb') as file:
            pkl.dump(self.results, file)
    
    def _save_image(self, uuid, images):
        if not os.path.exists(f"./TEMP/{uuid}"):
            os.makedirs(f"./TEMP/{uuid}")
        for i in range(len(images)):
            cv2.imwrite(f"./TEMP/{uuid}/{i}.jpg", images[i])

    def update_results(self, uuid, item_results):
        vehicle_actual = item_results.copy()
        self.results[uuid] = vehicle_actual
        self._sync_results()

    def queue_images(self, uuid, images):
        self.image_uuid.append(uuid)
        self._sync_image_uuid()
        self._save_image(uuid, images)
    
    def dequeue(self, inference_mode=True, variant="", suffix_no="", frame_no=""):
        result = None
        uuid_image = None
        try:
            uuid_image = self.image_uuid.pop(0)
            self._sync_image_uuid()

            if inference_mode:
                result = self.results.pop(uuid_image)
                self._sync_results()

            path = './TEMP/' + uuid_image + '/'
            try:
                directory = os.walk(f"./TEMP/{uuid_image}")
                for root, dirs, files in directory:
                    for file in files:
                        image = cv2.imread(f"./TEMP/{uuid_image}/{file}")
                        cam_name = file.split(".")[0]
                        saveImageCapture(image, variant=variant, suffix_no=suffix_no, camera_idx=cam_name)

                shutil.rmtree(path, ignore_errors=False, onerror=None)
                print("dequeue successed")
            except OSError as x:
                print("Error dequeue occured: %s : %s" % (path, x.strerror))
            
        except Exception as e:
            print(e)
            print("No image to dequeue")
            
        return result, uuid_image
    
    def delete(self, idx, inference_mode = True):
        result = None
        uuid_image = None
        try:
            uuid_image = self.image_uuid.pop(idx)
            self._sync_image_uuid()

            if inference_mode:
                result = self.results.pop(uuid_image)
                self._sync_results()

            path = './TEMP/' + uuid_image + '/'
            try:
                shutil.rmtree(path, ignore_errors=False, onerror=None)
                print("delete successed")
            except OSError as x:
                print("Error delete occured: %s : %s" % (path, x.strerror))
        except Exception as e:
            print(e)
            print("No image to delete")

        return result, uuid_image

    def get_image(self, uuid, cam_id):
        return cv2.imread(f"TEMP/{uuid}/{cam_id}.jpg")
    
    def get_image_length(self):
        return len(self.image_uuid)
    
    def get_image_uuid(self):
        return self.image_uuid

    def get_results(self):
        return self.results

