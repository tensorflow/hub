# TensorFlow Hub

## Publishing on tfhub.dev

Are you interested in publishing your modules on tfhub.dev? Express your
interest via our [Publisher
Survey](https://services.google.com/fb/forms/tensorflowhubpublishersurvey/). We
appreciate your valuable feedback, and will be providing more information about
publishing modules in the coming months. For now, please read our documentation
about [Hosting a Module](hosting.md).

## Fairness

As in all of machine learning, [fairness](http://ml-fairness.com) is an
[important](https://research.googleblog.com/2016/10/equality-of-opportunity-in-machine.html)
consideration. Modules typically leverage large pretrained datasets. When
reusing such a dataset, it’s important to be mindful of what data it contains
(and whether there are any existing biases there), and how these might impact
your downstream experiments.


## Status

Although we hope to prevent breaking changes, this project is still under active
development and is not yet guaranteed to have a stable API or module format.


## Security

Since they contain arbitrary TensorFlow graphs, modules can be thought of as
programs. [Using TensorFlow Securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md)
describes the security implications of referencing a module from an untrusted
source.


## Source-Code & Bug Reports

The source code is available on [GitHub](https://github.com/tensorflow/hub).
Use [GitHub issues](https://github.com/tensorflow/hub/issues) for feature requests
and bugs. Please see the [TensorFlow Hub mailing
list](https://groups.google.com/a/tensorflow.org/forum/#!forum/hub) for general
questions and discussion.
