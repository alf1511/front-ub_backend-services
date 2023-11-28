import cv2

from ultralytics.utils.plotting import Annotator, colors

class Camera:
    def __init__(self, index, path, cv_backend):
        self.index = index
        self.path = path

        self.cv_backend = cv_backend

        self.original_frame = None

        self._initialize_camera()
    
    def _initialize_camera(self):
        if self.cv_backend == "v4l2":
            self.camera = cv2.VideoCapture(self.path, cv2.CAP_V4L2)
        elif self.cv_backend == "dshow":
            self.camera = cv2.VideoCapture(self.path, cv2.CAP_DSHOW)
        else:
            self.camera = cv2.VideoCapture(self.path)

    def _draw_label(self, frame, result):
        for r in result:
            annotator = Annotator(frame, line_width=3)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                self._update_results(self.model.names[int(c)], box.conf[0].item())
                annotator.box_label(
                    b,
                    self.model.names[int(c)] + str(round(box.conf[0].item(), 2)),
                    color=colors(c, True),
                )

    def get_frame(self):
        try:
            if not self.isOpened():
                self._initialize_camera()
            success, frame = self.camera.read()
            if success:
               self.original_frame = frame
            return success, frame 
        except Exception as e:
            print(e)
            raise Exception(e)

    def get_result(self):
        return self.detect_results
    
    def release_camera(self):
        self.camera.release()

    def isOpened(self):
        return self.camera.isOpened()

    def reset(self):
        self.original_frame = None
