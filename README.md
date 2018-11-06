# Image Segmentation : Implementation of UNet and seg_depth models.

Image Segmentation : Implementation of UNet and seg_depth models in tf.keras.


<p align="center">
  <img src="https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/FCN1.png" width="50%" >
</p>


## Models 

* U-Net_tf-keras
* Seg_depth_tf-keras

## Getting Started

### Prerequisites

* Keras 2.0
* pandas
* tensorflow r1.11
* opencv for python

```shell
sudo apt-get install python-opencv
sudo apt-get install opencv-python
sudo pip install --upgrade tensorflow
sudo pip install --upgrade keras
```

### Preparing the data for training

You need to make two folders

*  Images Folder - For all the training images 
* Seg Folder - For the corresponding ground truth segmentation images
* Depth Folder - For the corresponding ground truth depth images


## Visualizing the prepared data

You can also visualize your prepared annotations for verification of the prepared data.

```shell
python visualizeDataset.py \
 --images="data/dataset1/images_prepped_train/" \
 --annotations="data/dataset1/annotations_prepped_train/" \

```

## Training the tf Model

To train the model run the following command:

```shell
python  tf_train_segdepth.py \
 --save_weights_path= "/home/xdjf/下载/image-segmentation-keras-tensorflow/" \
 --images_path = "data/dataset1/images_prepped_train/" \
 --seg_path =  \
 --depth_path =  \
 --input_shape = [480,160] \
 --epochs = 1 \
 --batch_size = 1
```

## Retrain the tf Model

To train the model run the following command:

```shell
python  tf_train_segdepth.py \
 --save_weights_path= "/home/xdjf/下载/image-segmentation-keras-tensorflow/" \
 --images_path = "data/dataset1/images_prepped_train/" \
 --seg_path =  \
 --depth_path =  \
 --input_shape = [480,160] \
 --epochs = 1 \
 --batch_size = 1 \
 --retrain = True
```

## Getting the predictions

To get the seg_depth predictions of a trained tf_model

```shell
python  seg_depth_predict.py \
 --save_weights_path= "/home/xdjf/下载/image-segmentation-keras-tensorflow/" \
 --images_path = "data/dataset1/images_prepped_train/" \
 --input_shape = [480,160]
```

