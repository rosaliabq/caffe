# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

## Custom Layers

This caffe distribution includes a combination of custom layers to be able to train and deploy different types of networks:

* **DenseImageData** : Data layer that allows to train semantic segmentation networks. It supports augmentation with synthetic data (batches generated with half-real, half-synthetic images). Adding extra support to pretrain the encoder with parameter "Scale" as in ENet paper (https://arxiv.org/abs/1606.02147) .Source: https://github.com/alexgkendall/caffe-segnet

```
layer {
  name: "data"
  type: "DenseImageData"
  top: "data"
  top: "label"
  dense_image_data_param {
    source: "/mnt/ssd/home/rosalia/data/cityscapes/train.txt"
    synth_source: "/mnt/ssd/home/rosalia/data/synthia/synth.txt"
    new_width: 480
    new_height: 300
    batch_size: 16
    shuffle: true
    scale: 1.0 #scale down label image to pretrain encoder
  }
}
```

* **Upsample** : This layer upsamples (unpooling operation) the input feature map according to some previous pooling operation. Source: https://github.com/alexgkendall/caffe-segnet

```
layer {
  name: "decoder_bottle1_other_unpool"
  type: "Upsample"
  bottom: "decoder_bottle1_other_bn"
  bottom: "bottleneck6_other_pool_mask"
  top: "decoder_bottle1_other_unpool"
  upsample_param {
    scale: 2
    pad_out_h: true     #Use padding in case of initial odd width or height
    pad_out_w: false
  }
}
```

* **SoftmaxWithLoss** : Added class weighting (it helps to deal with imbalanced datasets). Source: https://github.com/alexgkendall/caffe-segnet

```
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "conv8"
  bottom: "label"
  top: "loss"
  loss_param {
    #ignore_label: 0
    weight_by_label_freqs: true
    class_weighting: 0.4
    class_weighting: 0.3
    class_weighting: 0.6
    class_weighting: 13.7
    class_weighting: 2.0
    class_weighting: 0.5
    class_weighting: 3.7
    class_weighting: 1.4
  }
  softmax_param {
    engine: CAFFE
  }
}
```

* **ConvolutionDepthwise** : Depthwise convolutions as implemented in MobileNet paper. Source: https://github.com/BVLC/caffe/pull/5665

```
layer {
  name: "conv1/dw"
  type: "ConvolutionDepthwise"
  bottom: "conv0"
  top: "conv1/dw"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 32
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
```
* **Interp** : Interpolation layer, allows to resize (shrink or zoom) an image by a factor. Source: https://github.com/hszhao/PSPNet

```
layer {
  name: "data_sub2"
  type: "Interp"
  bottom: "data_sub1"
  top: "data_sub2"
  interp_param {
    shrink_factor: 2
    //zoom_factor: 2
  }
}
```

* **YOLO support**: adding support for YOLOv1. Source: https://github.com/yeahkun/caffe-yolo

**BoxData**: support for YOLO data input encoding. 

```
layer {
  name: "data"
  type: "BoxData"
  top: "data"
  top: "label"
  include {
    phase: TRAIN  
  }
  transform_param {
    mirror: false
    force_color: true
    mean_value: 155
    mean_value: 155
    mean_value: 155
  }
  data_param {
    source: ".pascalvoc/trainval_lmdb"
    batch_size: 16
    side: 7
    backend: LMDB
  }
}
```

**DetectionLoss**: YOLO loss function.

```
layer {
  name: "loss"
  type: "DetectionLoss"
  bottom: "regression"
  bottom: "label"
  top: "loss"
  loss_weight: 1
  detection_loss_param {
    side: 7
    num_class: 20
    num_object: 2
    object_scale: 1.0
    noobject_scale: 0.5
    class_scale: 1.0
    coord_scale: 5.0
    sqrt: true
    constriant: true
  }
}
```

**New lr policy**: "multifixed" (configurable with stagelr+stageiter)

* **Bounded ReLU support**: works as a ReLU, with a upper limit (if x<0: f(x)=0, if 0<x<bound: f(x)=x, if x>bound: f(x)=bound).

``` 
layer {
  name: "conv1/relu"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
  relu_param {
    bound: 15
  }
}
```

## Custom distributions

 - [Intel Caffe](https://github.com/BVLC/caffe/tree/intel) (Optimized for CPU and support for multi-node), in particular Xeon processors (HSW, BDW, SKX, Xeon Phi).
- [OpenCL Caffe](https://github.com/BVLC/caffe/tree/opencl) e.g. for AMD or Intel devices.
- [Windows Caffe](https://github.com/BVLC/caffe/tree/windows)

## Community

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BAIR/BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
