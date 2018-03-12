import sys
sys.path.append("../classifier")
import classifier

from detection_process import DetectionProcess
from heat_map import HeatMap
from detector import Detector
from search_windows_factory import SearchWindowsFactory
from sklearn.externals import joblib


car_id = 1 # TODO: grab from saved model
height, width = 720, 1280

image0 = "../../input/bbox-example-image.jpg"
image1 = "../../input/img1.png"
image2 = "../../input/991.png"
image3 = "../../input/275.png"
image4 = "../../input/600.png"
image5 = "../../input/750.png"
image6 = "../../input/1025.png"
image7 = "../../input/1150.png"
image8 = "../../input/1250.png"
video0 = "../../input/project_video.mp4"

configs = [
    {"x_start_stop":(None, None), "y_start_stop":(390, 520), "xy_window":(64, 64), "xy_overlap":(0.75, 0.75)},
    {"x_start_stop":(None, None), "y_start_stop":(380, 650), "xy_window":(128, 128), "xy_overlap":(0.5, 0.5)},
    {"x_start_stop":(None, None), "y_start_stop":(380, 660), "xy_window":(168, 168), "xy_overlap":(0.5, 0.75)},
    {"x_start_stop":(None, None), "y_start_stop":(400, 660), "xy_window":(256, 256), "xy_overlap":(0.5, 0.5)},
]

search_windows_factory = SearchWindowsFactory()
search_windows = search_windows_factory.create(height, width, configs)
print("Total amount of search windows is {}.".format(len(search_windows)))

clf = joblib.load("../training_results/model.pkl")

heat_map = HeatMap(height, width, threshold=2)
vehicles_detector = Detector(clf, search_windows, heat_map, car_id)
process = DetectionProcess(vehicles_detector.detect)

# process.run_on_image(image8)
process.run_on_video(video0)
