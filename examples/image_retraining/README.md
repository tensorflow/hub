`retrain.py` is an example script that shows how one can adapt a pretrained
network for other classification problems. A detailed overview of this script
can be found at: https://www.tensorflow.org/hub/tutorials/image_retraining

The script also shows how one can train layers
with quantized weights and activations instead of taking a pre-trained floating
point model and then quantizing weights and activations.
The output `GraphDef` produced by this script is compatible with the TensorFlow
Lite Optimizing Converter and can be converted to TFLite format.
