# Image Modules

For image classification, two kinds of modules are available:

  * Modules to do [image
    classification](../common_signatures/images.md#classification)
    with the particular classes that the module has been trained for.
  * Modules to extract [image feature
    vectors](../common_signatures/images.md#feature-vector),
    (a.k.a. "bottleneck values") for use in custom image classifiers.
    (This is elaborated in the [image retraining
    tutorial](../tutorials/image_retraining.md).)

Click on a module to view its documentation, or reference the URL from the
TensorFlow Hub library like so:

```python
m = hub.Module("https://tfhub.dev/...")
```


## Modules trained on ImageNet (ILSVRC-2012-CLS)

### Inception and Inception-ResNet

  * Inception V1:
    [classification](https://tfhub.dev/google/imagenet/inception_v1/classification/1),
    [feature_vector](https://tfhub.dev/google/imagenet/inception_v1/feature_vector/1).
  * Inception V2:
    [classification](https://tfhub.dev/google/imagenet/inception_v2/classification/1),
    [feature_vector](https://tfhub.dev/google/imagenet/inception_v2/feature_vector/1).
  * Inception V3:
    [classification](https://tfhub.dev/google/imagenet/inception_v3/classification/1),
    [feature_vector](https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1).
  * Inception-ResNet V2:
    [classification](https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/1),
    [feature_vector](https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1).


### MobileNet

MobileNets come in various sizes controlled by a multiplier for the depth
(number of features), and trained for various sizes of input images.
See the module documentation for details.

  * MobileNet V1

    |          | 224x224 | 192x192 | 160x160 | 128x128 |
    |----------|---------|---------|---------|---------|
    | **100%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/1) |
    |  **75%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/feature_vector/1) |
    |  **50%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/1) |
    |  **25%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/1) |

  * MobileNet V1 instrumented for quantization with TF-Lite ("/quantops")

    |          | 224x224 | 192x192 | 160x160 | 128x128 |
    |----------|---------|---------|---------|---------|
    | **100%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/quantops/feature_vector/1) |
    |  **75%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/quantops/feature_vector/1) |
    |  **50%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/quantops/feature_vector/1) |
    |  **25%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/quantops/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/quantops/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/quantops/feature_vector/1) |

  * MobileNet V2

    |          | 224x224 | 192x192 | 160x160 | 128x128 | 96x96 |
    |----------|---------|---------|---------|---------|-------|
    | **140%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2) | | | | |
    | **130%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/2) | | | | |
    | **100%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/2) |
    |  **75%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_075_192/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_075_192/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_075_160/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_075_160/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_075_128/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_075_128/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/2) |
    |  **50%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/2) |
    |  **35%** | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_035_192/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_035_192/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_035_160/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_035_160/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/2) | [classification](https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/classification/2)<br/>[feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/feature_vector/2) |


### NASNet and PNASNet

  * NASNet-A large:
    [classification](https://tfhub.dev/google/imagenet/nasnet_large/classification/1),
    [feature_vector](https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1).
  * NASNet-A mobile:
    [classification](https://tfhub.dev/google/imagenet/nasnet_mobile/classification/1),
    [feature_vector](https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/1).
  * PNASNet-5 large:
    [classification](https://tfhub.dev/google/imagenet/pnasnet_large/classification/2),
    [feature_vector](https://tfhub.dev/google/imagenet/pnasnet_large/feature_vector/2).


### ResNet

  * ResNet V1

    | 50 layers | 101 layers | 152 layers |
    |-----------|------------|------------|
    | [classification](https://tfhub.dev/google/imagenet/resnet_v1_50/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/resnet_v1_101/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/resnet_v1_152/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/1) |

  * ResNet V2

    | 50 layers | 101 layers | 152 layers |
    |-----------|------------|------------|
    | [classification](https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/resnet_v2_101/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/1) | [classification](https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1)<br/>[feature_vector](https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1) |


## Modules trained on domain-specific datasets

### iNaturalist (iNat) 2017

The iNat2017 dataset consists of 579,184 training images and 95,986 validation
images from 5,089 species, taken from
[www.inaturalist.org](http://www.inaturalist.org).

  * Inception V3:
    [feature_vector](https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/1).
