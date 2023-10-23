from concurrent.futures import ThreadPoolExecutor
import threading

class CameraController:
    def __init__(self, cameras, inference_mode, release_cam):
        self.cameras = cameras
        self.inference_mode = inference_mode
        self.release_cam = release_cam
        self.threads = []
        self.results = {}
        self.images = []
        self.lock = threading.Lock()

    def capture_images(self, max_threads=5):
        try:
            self._start_capture(max_threads)
        except Exception as e:
            print(e)
        finally:
            self._stop_capture()
            print('Capturing process done.')
    
    def get_results(self):
        with self.lock:
            return self.results
    
    def get_images(self):
        with self.lock:
            return self.images

    def _start_capture(self, max_threads):
        with ThreadPoolExecutor(max_threads) as executor:
            camera_ids = list(self.cameras.keys())
            executor.map(self._capture_image, camera_ids, [self.release_cam] * len(camera_ids))

    def _stop_capture(self):
        for thread in self.threads:
            thread.join()

    def _capture_image(self, camera_id, release_cam):
        self.cameras[camera_id].get_frame()
        with self.lock:
            self.images.append(self.cameras[camera_id].original_frame)

            if self.inference_mode:
                self.cameras[camera_id].inference_frame(False)
                results = self.cameras[camera_id].get_result()

                for r in results:
                    if r not in self.results:
                        self.results[r] = results[r]
                    else:
                        self.results[r] = max(self.results[r], results[r])

                self.cameras[camera_id].reset()
                if release_cam:
                    self.cameras[camera_id].release_camera()
    
    def reset_images_results(self):
        with self.lock:
            self.results = {}
            self.images = []
