# Unsupervised Visual Recognition by Solving Arbitrary Puzzles
A Tensorflow implementation for the paper:

Iterative Reorganization with Weak Spatial Constraints:<br>
Solving Arbitrary Jigsaw Puzzles for Unsupervised Representation Learning<br>
[Chen Wei](https://weichen582.github.io/), [Lingxi Xie](http://lingxixie.com/), [Xutong Ren](https://tonghelen.github.io/), [Yingda Xia](http://yingdaxia.github.io/), Chi Su, [Jiaying Liu](http://www.icst.pku.edu.cn/struct/people/liujiaying.html), [Qi Tian](http://www.cs.utsa.edu/~qitian/), [Alan L. Yuille](http://cs.jhu.edu/~ayuille/)<br>

[Paper (hosted by arXiv)](https://arxiv.org/abs/1812.00329), [Project Page]()

<img src="figs/framework.png" width="800px"/>

### Requirements ###
1. Python3
2. Tensorflow >= 1.8.0
3. numpy

### Usage ###
#### Dataset ####
Please download [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/) dataset first. Save the files recording paths and labels of training and validation data at some place, just like `files/val_sub_cls.txt`. `files/val_sub_cls.txt` is a small subset of validation data of ILSVRC2012, which we use for fast validation during the training of the puzzle models. Remember to set the **train_paths_file** flag and the **val_paths_file** flag for training and validation/testing data, respectively.

#### Backbone ####
This code base supports AlexNet, ResNet18 (v1) and ResNet50 (v1) as the backbone for feature extraction. Use the **backbone** flag for different architectures.

#### Settings ####
As mentioned in the paper, we equip our models with *unary term*, *binary term* and *mirror augmentatoin*. Each model has unary terms to predict the arbitrary jigsaw puzzles. To turn on binary temrs, set the **binary** flag. To add mirror augmentation, *i.e.*, random left-right flip, include **flip_lr** in the **preprocess** flag.

#### Config ####
`config` contains several shell scripts for you to refer to, which are to train or evaluate a model for solving arbitrary jigsaw puzzles. The **experiment_name** flag is necessary to indicate a specific experiment.

#### Transfer Learning ####
We use this [VOC-Classification Repo](https://github.com/jeffdonahue/voc-classification) to evaluate our models on V0C2007 classification task. We use this [Fast R-CNN Repo](https://github.com/rbgirshick/fast-rcnn) to evalute our models on VOC2007 detection task. The tensorflow models are converted to caffe models to fit into these repos.
 
### Citation ###
 ```
 @inproceedings{chen2019iterative,
  title={Iterative Reorganization with Weak Spatial Constraints: Solving Arbitrary Jigsaw Puzzles for Unsupervised Representation Learning},
  author={Chen Wei, Lingxi Xie, Xutong Ren, Yingda Xia, Chi Su, Jiaying Liu, Qi Tian, Alan L. Yuille},
  booktitle={Computer Vision and Pattern Recognition},
  year={2019}
}
```

Some codes are based on [Revisiting-SSL repo](https://github.com/google/revisiting-self-supervised), which provides some insightful experimental results for Self-Supervised Learning.
 
