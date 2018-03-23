# Image Modules

For image classification, two kinds of modules are available:

  * Modules to do [image
    classification](../common_signatures/images.md#image-classification)
    with the particular classes that the module has been trained for.
  * Modules to extract [image feature
    vectors](../common_signatures/images.md#image-feature-vector),
    (a.k.a. "bottleneck values") for use in custom image classifiers.
    (This is elaborated in the [image retraining
    tutorial](../tutorials/image_retraining.md).)

The "module" links from the list below can be copy&pasted for use in
Python code like:

```python
m = hub.Module("https://tfhub.dev/...")
```

The "doc" links provide documentation, including an explanation of
module names and more detailed usage instructions.


## Modules trained on ImageNet (ILSVRC-2012-CLS)

### Inception and Inception-ResNet

  * **inception_v1**:
      classification: [doc](google/imagenet/inception_v1/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/inception_v1/classification/1);
      feature_vector: [doc](google/imagenet/inception_v1/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/inception_v1/feature_vector/1).
  * **inception_v2**:
      classification: [doc](google/imagenet/inception_v2/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/inception_v2/classification/1);
      feature_vector: [doc](google/imagenet/inception_v2/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/inception_v2/feature_vector/1).
  * **inception_v3**:
      classification: [doc](google/imagenet/inception_v3/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/inception_v3/classification/1);
      feature_vector: [doc](google/imagenet/inception_v3/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1).
  * **inception_resnet_v2**:
      classification: [doc](google/imagenet/inception_resnet_v2/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/1);
      feature_vector: [doc](google/imagenet/inception_resnet_v2/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1).


### MobileNet V1

Listed by decreasing size and number of operations.
See module docs for more explanations.

  * **mobilenet_v1_100_224**:
      classification: [doc](google/imagenet/mobilenet_v1_100_224/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_100_224/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1).
  * **mobilenet_v1_100_192**:
      classification: [doc](google/imagenet/mobilenet_v1_100_192/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_100_192/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/feature_vector/1).
  * **mobilenet_v1_100_160**:
      classification: [doc](google/imagenet/mobilenet_v1_100_160/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_100_160/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/feature_vector/1).
  * **mobilenet_v1_100_128**:
      classification: [doc](google/imagenet/mobilenet_v1_100_128/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_100_128/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/feature_vector/1).
  * **mobilenet_v1_075_224**:
      classification: [doc](google/imagenet/mobilenet_v1_075_224/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_075_224/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/feature_vector/1).
  * **mobilenet_v1_075_192**:
      classification: [doc](google/imagenet/mobilenet_v1_075_192/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_075_192/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/feature_vector/1).
  * **mobilenet_v1_075_160**:
      classification: [doc](google/imagenet/mobilenet_v1_075_160/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_075_160/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/feature_vector/1).
  * **mobilenet_v1_075_128**:
      classification: [doc](google/imagenet/mobilenet_v1_075_128/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_075_128/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/feature_vector/1).
  * **mobilenet_v1_050_224**:
      classification: [doc](google/imagenet/mobilenet_v1_050_224/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_050_224/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/feature_vector/1).
  * **mobilenet_v1_050_192**:
      classification: [doc](google/imagenet/mobilenet_v1_050_192/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_050_192/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/feature_vector/1).
  * **mobilenet_v1_050_160**:
      classification: [doc](google/imagenet/mobilenet_v1_050_160/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_050_160/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/feature_vector/1).
  * **mobilenet_v1_050_128**:
      classification: [doc](google/imagenet/mobilenet_v1_050_128/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_050_128/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/feature_vector/1).
  * **mobilenet_v1_025_224**:
      classification: [doc](google/imagenet/mobilenet_v1_025_224/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_025_224/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/feature_vector/1).
  * **mobilenet_v1_025_192**:
      classification: [doc](google/imagenet/mobilenet_v1_025_192/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_025_192/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/feature_vector/1).
  * **mobilenet_v1_025_160**:
      classification: [doc](google/imagenet/mobilenet_v1_025_160/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_025_160/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/feature_vector/1).
  * **mobilenet_v1_025_128**:
      classification: [doc](google/imagenet/mobilenet_v1_025_128/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_025_128/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/feature_vector/1).

#### Instrumented for quantization with TF-Lite

  * **mobilenet_v1_100_224/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_100_224/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/1).
  * **mobilenet_v1_100_192/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_100_192/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_100_192/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_192/quantops/feature_vector/1).
  * **mobilenet_v1_100_160/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_100_160/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_100_160/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_160/quantops/feature_vector/1).
  * **mobilenet_v1_100_128/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_100_128/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_100_128/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_100_128/quantops/feature_vector/1).
  * **mobilenet_v1_075_224/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_075_224/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_075_224/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_224/quantops/feature_vector/1).
  * **mobilenet_v1_075_192/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_075_192/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_075_192/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_192/quantops/feature_vector/1).
  * **mobilenet_v1_075_160/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_075_160/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_075_160/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_160/quantops/feature_vector/1).
  * **mobilenet_v1_075_128/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_075_128/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_075_128/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_075_128/quantops/feature_vector/1).
  * **mobilenet_v1_050_224/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_050_224/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_050_224/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_224/quantops/feature_vector/1).
  * **mobilenet_v1_050_192/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_050_192/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_050_192/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_192/quantops/feature_vector/1).
  * **mobilenet_v1_050_160/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_050_160/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_050_160/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_160/quantops/feature_vector/1).
  * **mobilenet_v1_050_128/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_050_128/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_050_128/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_050_128/quantops/feature_vector/1).
  * **mobilenet_v1_025_224/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_025_224/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_025_224/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_224/quantops/feature_vector/1).
  * **mobilenet_v1_025_192/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_025_192/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_025_192/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_192/quantops/feature_vector/1).
  * **mobilenet_v1_025_160/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_025_160/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_025_160/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_160/quantops/feature_vector/1).
  * **mobilenet_v1_025_128/quantops**:
      classification: [doc](google/imagenet/mobilenet_v1_025_128/quantops/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/quantops/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v1_025_128/quantops/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v1_025_128/quantops/feature_vector/1).


### MobileNet V2

Listed by decreasing size and number of operations.
See module docs for more explanations.

  * **mobilenet_v2_140_224**:
      classification: [doc](google/imagenet/mobilenet_v2_140_224/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_140_224/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/1).
  * **mobilenet_v2_130_224**:
      classification: [doc](google/imagenet/mobilenet_v2_130_224/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_130_224/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/feature_vector/1).
  * **mobilenet_v2_100_224**:
      classification: [doc](google/imagenet/mobilenet_v2_100_224/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_100_224/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1).
  * **mobilenet_v2_100_192**:
      classification: [doc](google/imagenet/mobilenet_v2_100_192/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_100_192/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/1).
  * **mobilenet_v2_100_160**:
      classification: [doc](google/imagenet/mobilenet_v2_100_160/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_100_160/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_100_160/feature_vector/1).
  * **mobilenet_v2_100_128**:
      classification: [doc](google/imagenet/mobilenet_v2_100_128/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_100_128/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_100_128/feature_vector/1).
  * **mobilenet_v2_100_96**:
      classification: [doc](google/imagenet/mobilenet_v2_100_96/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_100_96/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/1).
  * **mobilenet_v2_075_224**:
      classification: [doc](google/imagenet/mobilenet_v2_075_224/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_075_224/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/feature_vector/1).
  * **mobilenet_v2_075_192**:
      classification: [doc](google/imagenet/mobilenet_v2_075_192/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_075_192/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_075_192/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_075_192/feature_vector/1).
  * **mobilenet_v2_075_160**:
      classification: [doc](google/imagenet/mobilenet_v2_075_160/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_075_160/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_075_160/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_075_160/feature_vector/1).
  * **mobilenet_v2_075_128**:
      classification: [doc](google/imagenet/mobilenet_v2_075_128/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_075_128/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_075_128/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_075_128/feature_vector/1).
  * **mobilenet_v2_075_96**:
      classification: [doc](google/imagenet/mobilenet_v2_075_96/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_075_96/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_075_96/feature_vector/1).
  * **mobilenet_v2_050_224**:
      classification: [doc](google/imagenet/mobilenet_v2_050_224/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_050_224/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/feature_vector/1).
  * **mobilenet_v2_050_192**:
      classification: [doc](google/imagenet/mobilenet_v2_050_192/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_050_192/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/feature_vector/1).
  * **mobilenet_v2_050_160**:
      classification: [doc](google/imagenet/mobilenet_v2_050_160/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_050_160/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/feature_vector/1).
  * **mobilenet_v2_050_128**:
      classification: [doc](google/imagenet/mobilenet_v2_050_128/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_050_128/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_050_128/feature_vector/1).
  * **mobilenet_v2_050_96**:
      classification: [doc](google/imagenet/mobilenet_v2_050_96/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_050_96/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_050_96/feature_vector/1).
  * **mobilenet_v2_035_224**:
      classification: [doc](google/imagenet/mobilenet_v2_035_224/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_035_224/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/1).
  * **mobilenet_v2_035_192**:
      classification: [doc](google/imagenet/mobilenet_v2_035_192/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_035_192/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_035_192/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_035_192/feature_vector/1).
  * **mobilenet_v2_035_160**:
      classification: [doc](google/imagenet/mobilenet_v2_035_160/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_035_160/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_035_160/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_035_160/feature_vector/1).
  * **mobilenet_v2_035_128**:
      classification: [doc](google/imagenet/mobilenet_v2_035_128/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_035_128/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_035_128/feature_vector/1).
  * **mobilenet_v2_035_96**:
      classification: [doc](google/imagenet/mobilenet_v2_035_96/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/classification/1);
      feature_vector: [doc](google/imagenet/mobilenet_v2_035_96/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/mobilenet_v2_035_96/feature_vector/1).


### NASNet

  * **nasnet_large**:
      classification: [doc](google/imagenet/nasnet_large/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/nasnet_large/classification/1);
      feature_vector: [doc](google/imagenet/nasnet_large/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/nasnet_large/feature_vector/1).
  * **nasnet_mobile**:
      classification: [doc](google/imagenet/nasnet_mobile/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/nasnet_mobile/classification/1);
      feature_vector: [doc](google/imagenet/nasnet_mobile/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/nasnet_mobile/feature_vector/1).


### ResNet

  * **resnet_v1_50**:
      classification: [doc](google/imagenet/resnet_v1_50/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v1_50/classification/1);
      feature_vector: [doc](google/imagenet/resnet_v1_50/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v1_50/feature_vector/1).
  * **resnet_v1_101**:
      classification: [doc](google/imagenet/resnet_v1_101/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v1_101/classification/1);
      feature_vector: [doc](google/imagenet/resnet_v1_101/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v1_101/feature_vector/1).
  * **resnet_v1_152**:
      classification: [doc](google/imagenet/resnet_v1_152/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v1_152/classification/1);
      feature_vector: [doc](google/imagenet/resnet_v1_152/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/1).
  * **resnet_v2_50**:
      classification: [doc](google/imagenet/resnet_v2_50/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1);
      feature_vector: [doc](google/imagenet/resnet_v2_50/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/1).
  * **resnet_v2_101**:
      classification: [doc](google/imagenet/resnet_v2_101/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v2_101/classification/1);
      feature_vector: [doc](google/imagenet/resnet_v2_101/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/1).
  * **resnet_v2_152**:
      classification: [doc](google/imagenet/resnet_v2_152/classification/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1);
      feature_vector: [doc](google/imagenet/resnet_v2_152/feature_vector/1.md),
      [module](https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1).
