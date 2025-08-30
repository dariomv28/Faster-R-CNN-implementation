Faster R-CNN Implementation in Pytorch
========

This repository implements [Faster R-CNN](https://arxiv.org/abs/1506.01497) with training, inference and map evaluation in PyTorch.
The aim was to create a simple implementation based on PyTorch faster r-cnn codebase and to get rid of all the abstractions and make the implementation easy to understand.

The implementation caters to batch size of 1 only and uses roi pooling on single scale feature map.
The repo is meant to train faster r-cnn on voc dataset. Specifically I trained on VOC 2007 dataset.
## Data preparation
For setting up the VOC 2007 dataset:
* Download VOC 2007 train/val data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and name it as `VOC2007` folder
* Download VOC 2007 test data from http://host.robots.ox.ac.uk/pascal/VOC/voc2007 and name it as `VOC2007-test` folder
* Place both the directories inside the root folder of repo according to below structure
    ```
    FasterRCNN-Pytorch
        -> VOC2007
            -> JPEGImages
            -> Annotations
        -> VOC2007-test
            -> JPEGImages
            -> Annotations
        -> tools
            -> train.py
            -> infer.py
            -> train_torchvision_frcnn.py
            -> infer_torchvision_frcnn.py
        -> config
            -> voc.yaml
        -> model
            -> faster_rcnn.py

## Differences from Faster RCNN paper
This repo has some differences from actual Faster RCNN paper.
* Caters to single batch size
* Uses a randomly initialized fc6 fc7 layer of 1024 dim.
* Most of the hyper-parameters have directly been picked from official version and have not been tuned to this setting of 1024 dimensional fc layers. As of now using this I am getting ~61-62% mAP.
* To improve the results one can try the following:
  * Use VGG fc6 and fc7 layers
  * Tune the weight of different losses
  * Experiment with roi batch size
  * Experiment with hard negative mining
        -> dataset
            -> voc.py
    ```
