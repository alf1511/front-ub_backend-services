from concurrent.futures import ThreadPoolExecutor

class CameraController:
    def __init__(self, cameras, inference_mode):
        self.cameras = cameras
        self.inference_mode = inference_mode
        self.threads = []
        self.results = {}
        self.images = []

    def capture_images(self, max_threads=5):
        try:
            self._start_capture(max_threads)
        except Exception as e:
            print(e)
        finally:
            self._stop_capture()
            print('Capturing process done.')
    
    def get_results(self):
        return self.results
    
    def get_images(self):
        return self.images

    def _start_capture(self, max_threads):
        with ThreadPoolExecutor(max_threads) as executor:
            camera_ids = list(self.cameras.keys())
            executor.map(self._capture_image, camera_ids)

    def _stop_capture(self):
        for thread in self.threads:
            thread.join()

    def _capture_image(self, camera_id):
        self.cameras[camera_id].get_frame()
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
            self.cameras[camera_id].release_camera()
    
    def reset_images_results(self):
        self.results = {}
        self.images = []