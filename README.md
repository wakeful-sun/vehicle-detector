**Vehicle Detection Project**

The project implements a pipeline that detects vehicles on video stream and marks them with a bounding boxes.

Project consists of two parts: classifier training pipeline and actual detector/tracker.

[//]: # (Image References)
[image1]: ./output_images/train_labels.png "Labels amount for each class in training set"
[image2]: ./output_images/test_labels.png "Labels amount for each class in testing set"
[image3]: ./output_images/labels_distribution.png "Labels distribution"
[image4]: ./output_images/random_car_images.png "Random car images"
[image5]: ./output_images/random_non_car_images.png "Random non-car images"
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
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
| trainig set | testing set |

![alt text][image3]

*Random images*

|![alt text][image4] |![alt text][image5]|
|:---:|:---:|
| car images | non car images |

#
###### Classifier training pipeline

Classifier and its [training pipeline](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/classifier.py#L33-L52) can be found in `.\code\classifier\` directory.

Before classifier fit each image goes through pre-processing pipeline:
 - [image resize](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/features_extractor.py#L13) to `32x32` shape
 - [conversion](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/features_extractor.py#L14) to "YUV" color space
 - [transformation](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/features_extractor.py#L15) into features vector (features extraction), that consists of Spatial Binning of Color features, Color Histogram features and Histogram of Oriented Gradients (HOG) features
 - [scaler fit](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/classifier.py#L45) and features vector [normalization](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/classifier.py#L46)

And actual [classifier fit](https://github.com/wakeful-sun/vehicle-detector/blob/b0d4d4edbedf5933e5a91f4264bea51c55856c60/code/classifier/classifier.py#L47).

When classifier training is done, program saves training summary information into text file, and saves trained model to `.\code\training_results\model.pkl` file with help of [`sklearn.externals.joblib`](http://scikit-learn.org/stable/modules/model_persistence.html). So model can be loaded and used later on in vehicle detection pipeline.

#
###### Histogram of Oriented Gradients (HOG)
HOG feature extraction is implemented in `.\code\classifier\features\hog.py` file.

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

`"YUV"` color scheme gives smaller overall accuracy, but more stable detection of black colored vehicles.


##### 2. Vehicle detection

Car detection pipeline code is located in `.\code\detection_pipeline\` directory and has entry point in `.\code\detection_pipeline\main.py` file.



##### 3. Video Implementation

Here's a [link to my video result](./output.mp4)


---

### Discussion

Sliding window search is slow. 
-->