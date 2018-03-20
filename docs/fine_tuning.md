# Fine-Tuning

Training the variables of an imported module together with those of the model
around it is called *fine-tuning*. Fine-tuning can result in better quality, but
adds new complications. We advise consumers to look into fine-tuning only after
exploring simpler quality tweaks.

## For Consumers

To enable fine-tuning, instantiate the module with
`hub.Module(..., trainable=True)` to make its variables trainable and
import TensorFlow's `REGULARIZATION_LOSSES`. If the module has multiple
graph variants, make sure to pick the one approprate for training.
Usually, that's the one with tags `{"train"}`.

Choose a training regime that does not ruin the pre-trained weights,
for example, a lower learning rate than for training from scratch.

## For Publishers

To make fine-tuning easier for consumers, please be mindful of the following:

*   Fine-tuning needs regularization. Your module is exported with the
    `REGULARIZATION_LOSSES` collection, which is what puts your choice of
    `tf.layers.dense(..., kernel_regularizer=...)` etc. into what the consumer
    gets from `tf.losses.get_regularization_losses()`. Prefer this way of
    defining L1/L2 regularization losses.

*   In the publisher model, avoid defining L1/L2 regularization via the `l1_`
    and `l2_regularization_strength` parameters of `tf.train.FtrlOptimizer`,
    `tf.train.ProximalGradientDescentOptimizer`, and other proximal
    optimizers. These are not exported alongside the module, and setting
    regularization strengths globally may not be appropriate for the
    consumer. Except for L1 regularization in wide (i.e. sparse linear) or wide
    & deep models, it should be possible to use individual regularization losses
    instead.

*   If you use dropout, batch normalization, or similar training techniques, set
    dropout rate and other hyperparameters to values that make sense across many
    expected uses.
