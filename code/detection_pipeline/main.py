from detection_process import DetectionProcess
from detector import Detector
from sliding_windows_factory import SlidingWindowsFactory


image1 = ""
video1 = ""

y_start_stop = (450, None)  # Min and max in y to search in slide_window()
xy_window = (64, 64)

sliding_windows_factory = SlidingWindowsFactory(x_start_stop=(None, None),
                                                y_start_stop=y_start_stop,
                                                xy_window=xy_window,
                                                xy_overlap=(0.5, 0.5))

clf = "load " #TODO: load classifier

vehicles_detector = Detector(data, clf, sliding_windows_factory)
process = DetectionProcess(vehicles_detector.detect)

process.run_on_image(image1)
# process.run_on_video(video1)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
# image = image.astype(np.float32)/255