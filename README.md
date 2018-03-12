**Vehicle Detection Project**

The project implements a pipeline that detects vehicles on video stream and marks them with a bounding boxes.

Project consists of two parts: classifier training pipeline and actual detector/tracker.

[//]: # (Image References)
[image1]: ./output_images/train_labels.png "Labels amount for each class in training set"
[image2]: ./output_images/test_labels.png "Labels amount for each class in testing set"
[image3]: ./output_images/labels_distribution.png "Labels distribution"
[image4]: ./output_images/random_car_images.png "Random car images"
[image5]: ./output_images/random_non_car_images.png "Random non-car images"

[car_example]: ./output_images/HOG/car.png "Car example"
[car_YUV_ch0]: ./output_images/HOG/YUV_channels/car_channel_0.png "Car YUV channel 0"
[car_YUV_ch1]: ./output_images/HOG/YUV_channels/car_channel_1.png "Car YUV channel 1"
[car_YUV_ch2]: ./output_images/HOG/YUV_channels/car_channel_2.png "Car YUV channel 2"
[car_HOG_ch0]: ./output_images/HOG/HOG_channels/car_hog_0.png "Car HOG channel 0"
[car_HOG_ch1]: ./output_images/HOG/HOG_channels/car_hog_1.png "Car HOG channel 1"
[car_HOG_ch2]: ./output_images/HOG/HOG_channels/car_hog_2.png "Car HOG channel 2"

[non_car_example]: ./output_images/HOG/non-car.png
[non_car_YUV_ch0]: ./output_images/HOG/YUV_channels/non_car_channel_0.png "Non car YUV channel 0"
[non_car_YUV_ch1]: ./output_images/HOG/YUV_channels/non_car_channel_1.png "Non car YUV channel 1"
[non_car_YUV_ch2]: ./output_images/HOG/YUV_channels/non_car_channel_2.png "Non car YUV channel 2"
[non_car_HOG_ch0]: ./output_images/HOG/HOG_channels/non_car_hog_0.png "Car HOG channel 0"
[non_car_HOG_ch1]: ./output_images/HOG/HOG_channels/non_car_hog_1.png "Car HOG channel 1"
[non_car_HOG_ch2]: ./output_images/HOG/HOG_channels/non_car_hog_2.png "Car HOG channel 2"

[all_search_windows]: ./output_images/all_search_windows.png "All search windows"

[video1]: ./project_video.mp4


#### 1. Classifier

 [<img align="left" alt="scikit learn logo" src="http://scikit-learn.org/stable/_static/scikit-learn-logo-small.png">](http://scikit-learn.org/stable/index.html) Car/non-car classification is built with help of Linear Support Vector Machine learning algorithm implementation from [**scikit-learn**](http://scikit-learn.org/stable/index.html) library - `LinearSVC` [linear Support Vector Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC).

 Classifier is trained on combined set of images from [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), provided by [**Udacity**](https://eu.udacity.com/). Here are the references ot [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images.

 Classifier entry point is located in `.\code\classifier\main.py` file.

#
###### Data exploration

Data for training and testing provided via [`DataProvider`](https://github.com/wakeful-sun/vehicle-detector/blob/master/code/classifier/data_provider.py) class, which also has some functionality for data exploration.

*Images amount for each class (car class - 1, non-car class - 0)*

|![alt text][image1] |![alt text][image2]|
|:---:|:---:|
| training set | testing set |

![alt text][image3]

*Random images*

|![alt text][image4] |![alt text][image5]|
|:---:|:---:|
| car images | non car images |

#
###### Classifier training pipeline

Classifier and its [training pipeline](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/classifier.py#L33-L52) can be found in `.\code\classifier\` directory.

Before classifier fit each image goes through pre-processing pipeline:
 - [image resize](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/features_extractor.py#L13) to `32x32` shape. It increased overall accuracy by 6% and decreased classifier decision time
 - [conversion](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/features_extractor.py#L14) to `YUV` color space. `YUV` color scheme gives smaller overall accuracy, but more stable detection of black colored vehicles. which makes me think about model over-fitting when RGB or HSV color scheme if used with training data set
 - [transformation](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/features_extractor.py#L15) into features vector (features extraction), that consists of **Spatial Binning of Color** features, **Color Histogram** features and **Histogram of Oriented Gradients (HOG)** features
 - [scaler fit](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/classifier.py#L45) and features vector [normalization](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/classifier.py#L46)

And actual [classifier fit](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/classifier.py#L47).

When classifier training is done, program saves training summary information into text file, and saves trained model to `.\code\training_results\model.pkl` file with help of [`sklearn.externals.joblib`](http://scikit-learn.org/stable/modules/model_persistence.html). So model can be loaded and used later on in vehicle detection pipeline.

#
###### Histogram of Oriented Gradients (HOG)
HOG feature extraction is implemented in `.\code\classifier\features\hog.py` file and uses `skimage.feature.hog` function to produce gradients.

*Car image*

|| color | channel 0 | channel 1 | channel 2 |
|:---:|:---:|:---:|:---:|:---:|
|YUV image|![alt text][car_example] |![alt text][car_YUV_ch0]|![alt text][car_YUV_ch1]|![alt text][car_YUV_ch2]|
|HOG image||![alt text][car_HOG_ch0]|![alt text][car_HOG_ch1]|![alt text][car_HOG_ch2]|

*Non-car image*

|| color | channel 0 | channel 1 | channel 2 |
|:---:|:---:|:---:|:---:|:---:|
|YUV image|![alt text][non_car_example] |![alt text][non_car_YUV_ch0]|![alt text][non_car_YUV_ch1]|![alt text][non_car_YUV_ch2]|
|HOG image||![alt text][non_car_HOG_ch0]|![alt text][non_car_HOG_ch1]|![alt text][non_car_HOG_ch2]|


HOG features factory uses next parameters:
``` python
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
```

`"ALL"` word tells `HistogramOfOrientedGradientsFeaturesFactory` to produce HOG feature vector for all image channels.
``` python
if self.hog_channel == "ALL":
    for channel in range(image.shape[2]):
        channel_features, _ = self._extract_channel_features(image, channel)
```


##### 2. Vehicle detection

Car detection pipeline code is located in `.\code\detection_pipeline\` directory and has entry point in `.\code\detection_pipeline\main.py` file.

Here is the [vehicle detection program](https://github.com/wakeful-sun/vehicle-detector/blob/master/code/detection_pipeline/main.py) points:
 - search windows creation. `SearchWindowsFactory` creates set of search window boundaries of different scales 
 - classifier restoration from dump file
 - heat map cache creation
 - detection process creation and execution

[Detection process](https://github.com/wakeful-sun/vehicle-detector/blob/8e19b021d516a2527862a51f3025a407d8fcf730/code/detection_pipeline/detector.py#L28-L40) does next:
 - [for each search window](https://github.com/wakeful-sun/vehicle-detector/blob/8e19b021d516a2527862a51f3025a407d8fcf730/code/detection_pipeline/detector.py#L17-L26) fetches image data in window boundaries and feeds the data to classificator. Car results are saved and passed to next step
 - [heat map](https://github.com/wakeful-sun/vehicle-detector/blob/8e19b021d516a2527862a51f3025a407d8fcf730/code/detection_pipeline/detector.py#L34) update. [`HeatMap`](https://github.com/wakeful-sun/vehicle-detector/blob/master/code/detection_pipeline/heat_map.py) class instance stores last 10 frames and responsible for `False` positives filtering
 - [fetching](https://github.com/wakeful-sun/vehicle-detector/blob/8e19b021d516a2527862a51f3025a407d8fcf730/code/detection_pipeline/detector.py#L35) confident cars boundaries
 - [drawing](https://github.com/wakeful-sun/vehicle-detector/blob/8e19b021d516a2527862a51f3025a407d8fcf730/code/detection_pipeline/detector.py#L36) cars bounding boxes on input image

###### Sliding search window

Current configuration has defines 493 search windows of different sizes.

|window size|windows amount |
|:---:|:---:|
|(64, 64)   |385            |
|(128, 128) |57             |
|(168, 168) |42             |
|(256, 256) |9              |

Search windows parametes are defined in dictionary:

``` python
configs = [
    {"x_start_stop": (None, None), "y_start_stop": (390, 520), "xy_window": (64, 64), "xy_overlap": (0.75, 0.75)},
    {"x_start_stop": (None, None), "y_start_stop": (380, 650), "xy_window": (128, 128), "xy_overlap": (0.5, 0.5)},
    {"x_start_stop": (None, None), "y_start_stop": (380, 660), "xy_window": (168, 168), "xy_overlap": (0.5, 0.75)},
    {"x_start_stop": (None, None), "y_start_stop": (400, 660), "xy_window": (256, 256), "xy_overlap": (0.5, 0.5)}
]
```
Then configuration passed to `` factory, that produces boundaries indexes for each window
``` python
search_windows = search_windows_factory.create(max_height=720, max_width=1280, configs=configs)
```

*Here is all search windows boundaries on one image*

![alt text][all_search_windows]

###### Heat map

Heap map is a nice concept that allows to produce confident car boundaries bounding box and get rid of false positive detections.
The main idea is to add some value (heat) to each area (search window) with possible car inside. Actual car position is likely to be detected in many search windows. 
After "heating" search windows indexes we apply threshold function that wipes out areas with small heat.
Non-zero indexes on resulting image are belong to car(s).

Another cool feature I used is [`scipy.ndimage.measurements.label`](https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.ndimage.measurements.label.html). 
This function accepts heat map image as parameter and labels each detached object on it with indexes.
In this way we can separate different cars and draw separate bounding boxes around each car.

My implementation also collects up to 10 heat maps from last frames and takes into account all of them when produces resulting car boundaries. That reduces false positives even more.

##### 3. Video Implementation

Here's a [link to my video result](./output.mp4)
