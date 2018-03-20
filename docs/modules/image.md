# Image Modules

These are [TensorFlow Hub modules](../../README.md) to extract
[image feature vectors](../common_signatures/images.md#image-feature-vector)
for use in custom image classifiers, and to do
[image classification](../common_signatures/images.md#image-classification)
with the original classes from the training dataset.

The "doc" links can be clicked to see documentation.
The "module" links can be copy&pasted for use in Python code like

```python
m = hub.Module("https://storage.googleapis.com/...")
```



## Trained on Imagenet (ILSVRC-2012-CLS)

### Inception and Inception-ResNet

  * **inception_v1**:
      classification: [doc](google/image/imagenet/inception_v1/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/inception_v1/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/inception_v1/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/inception_v1/feature_vector/1.tar.gz).
  * **inception_v2**:
      classification: [doc](google/image/imagenet/inception_v2/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/inception_v2/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/inception_v2/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/inception_v2/feature_vector/1.tar.gz).
  * **inception_v3**:
      classification: [doc](google/image/imagenet/inception_v3/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/inception_v3/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/inception_v3/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/inception_v3/feature_vector/1.tar.gz).
  * **inception_resnet_v2**:
      classification: [doc](google/image/imagenet/inception_resnet_v2/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/inception_resnet_v2/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/inception_resnet_v2/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/inception_resnet_v2/feature_vector/1.tar.gz).

### MobileNet V1

The standard size is mobilenet_v1_100_224 (listed on top).

  * **mobilenet_v1_100_224**:
      classification: [doc](google/image/imagenet/mobilenet_v1_100_224/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_100_224/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_100_224/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_100_224/feature_vector/1.tar.gz).
  * **mobilenet_v1_100_192**:
      classification: [doc](google/image/imagenet/mobilenet_v1_100_192/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_100_192/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_100_192/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_100_192/feature_vector/1.tar.gz).
  * **mobilenet_v1_100_160**:
      classification: [doc](google/image/imagenet/mobilenet_v1_100_160/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_100_160/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_100_160/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_100_160/feature_vector/1.tar.gz).
  * **mobilenet_v1_100_128**:
      classification: [doc](google/image/imagenet/mobilenet_v1_100_128/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_100_128/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_100_128/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_100_128/feature_vector/1.tar.gz).
  * **mobilenet_v1_075_224**:
      classification: [doc](google/image/imagenet/mobilenet_v1_075_224/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_075_224/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_075_224/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_075_224/feature_vector/1.tar.gz).
  * **mobilenet_v1_075_192**:
      classification: [doc](google/image/imagenet/mobilenet_v1_075_192/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_075_192/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_075_192/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_075_192/feature_vector/1.tar.gz).
  * **mobilenet_v1_075_160**:
      classification: [doc](google/image/imagenet/mobilenet_v1_075_160/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_075_160/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_075_160/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_075_160/feature_vector/1.tar.gz).
  * **mobilenet_v1_075_128**:
      classification: [doc](google/image/imagenet/mobilenet_v1_075_128/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_075_128/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_075_128/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_075_128/feature_vector/1.tar.gz).
  * **mobilenet_v1_050_224**:
      classification: [doc](google/image/imagenet/mobilenet_v1_050_224/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_050_224/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_050_224/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_050_224/feature_vector/1.tar.gz).
  * **mobilenet_v1_050_192**:
      classification: [doc](google/image/imagenet/mobilenet_v1_050_192/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_050_192/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_050_192/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_050_192/feature_vector/1.tar.gz).
  * **mobilenet_v1_050_160**:
      classification: [doc](google/image/imagenet/mobilenet_v1_050_160/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_050_160/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_050_160/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_050_160/feature_vector/1.tar.gz).
  * **mobilenet_v1_050_128**:
      classification: [doc](google/image/imagenet/mobilenet_v1_050_128/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_050_128/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_050_128/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_050_128/feature_vector/1.tar.gz).
  * **mobilenet_v1_025_224**:
      classification: [doc](google/image/imagenet/mobilenet_v1_025_224/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_025_224/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_025_224/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_025_224/feature_vector/1.tar.gz).
  * **mobilenet_v1_025_192**:
      classification: [doc](google/image/imagenet/mobilenet_v1_025_192/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_025_192/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_025_192/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_025_192/feature_vector/1.tar.gz).
  * **mobilenet_v1_025_160**:
      classification: [doc](google/image/imagenet/mobilenet_v1_025_160/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_025_160/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_025_160/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_025_160/feature_vector/1.tar.gz).
  * **mobilenet_v1_025_128**:
      classification: [doc](google/image/imagenet/mobilenet_v1_025_128/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_025_128/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/mobilenet_v1_025_128/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/mobilenet_v1_025_128/feature_vector/1.tar.gz).


### NASNet

  * **nasnet_large**:
      classification: [doc](google/image/imagenet/nasnet_large/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/nasnet_large/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/nasnet_large/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/nasnet_large/feature_vector/1.tar.gz).
  * **nasnet_mobile**:
      classification: [doc](google/image/imagenet/nasnet_mobile/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/nasnet_mobile/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/nasnet_mobile/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/nasnet_mobile/feature_vector/1.tar.gz).


### ResNet

  * **resnet_v1_50**:
      classification: [doc](google/image/imagenet/resnet_v1_50/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v1_50/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/resnet_v1_50/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v1_50/feature_vector/1.tar.gz).
  * **resnet_v1_101**:
      classification: [doc](google/image/imagenet/resnet_v1_101/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v1_101/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/resnet_v1_101/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v1_101/feature_vector/1.tar.gz).
  * **resnet_v1_152**:
      classification: [doc](google/image/imagenet/resnet_v1_152/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v1_152/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/resnet_v1_152/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v1_152/feature_vector/1.tar.gz).
  * **resnet_v2_50**:
      classification: [doc](google/image/imagenet/resnet_v2_50/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v2_50/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/resnet_v2_50/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v2_50/feature_vector/1.tar.gz).
  * **resnet_v2_101**:
      classification: [doc](google/image/imagenet/resnet_v2_101/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v2_101/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/resnet_v2_101/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v2_101/feature_vector/1.tar.gz).
  * **resnet_v2_152**:
      classification: [doc](google/image/imagenet/resnet_v2_152/classification/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v2_152/classification/1.tar.gz);
      feature_vector: [doc](google/image/imagenet/resnet_v2_152/feature_vector/1.md),
      [module](https://storage.googleapis.com/tfhub-test-modules/google/image/imagenet/resnet_v2_152/feature_vector/1.tar.gz).
