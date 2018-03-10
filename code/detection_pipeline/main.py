import sys
sys.path.append("../classifier")
import classifier

from detection_process import DetectionProcess
from detector import Detector
from sliding_windows_factory import SlidingWindowsFactory
from sklearn.externals import joblib


car_id = 1 # TODO: grab from saved model

image0 = "../../input/bbox-example-image.jpg"
video0 = "../../input/project_video.mp4"

y_start_stop = (450, None)  # Min and max in y to search
xy_window = (96, 96)

sliding_windows_factory = SlidingWindowsFactory(x_start_stop=(None, None),
                                                y_start_stop=y_start_stop,
                                                xy_window=xy_window,
                                                xy_overlap=(0.5, 0.5))

clf = joblib.load("../training_results/model.pkl")

vehicles_detector = Detector(clf, sliding_windows_factory, car_id)
process = DetectionProcess(vehicles_detector.detect)

process.run_on_image(image0)
# process.run_on_video(video0)
