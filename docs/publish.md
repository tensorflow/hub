<!--* freshness: { owner: 'maringeo' } *-->

# Become a publisher

## Terms of service

By submitting a model for publication, you agree to the TensorFlow Hub Terms of
Service at [https://tfhub.dev/terms](https://tfhub.dev/terms).

## Overview of the publishing process

The full process of publishing consists of:

1.  Creating the model (see how to
    [export a model](exporting_tf2_saved_model.md))
1.  Writing documentation (see how to
    [write model documentation](writing_model_documentation.md)
1.  Creating a publishing request (see how to
    [contribute](contribute_a_model.md))

## Publisher page specific markdown format

Publisher documentation is declared in the same kind of markdown files as
described in the [writing model documentation](writing_model_documentation)
guide, with slight syntactic differences.

The correct location for the publisher file on the TensorFlow Hub repo is:
[hub/tfhub_dev/assets/](https://github.com/tensorflow/hub/tree/master/tfhub_dev/assets)/<publisher_name>/<publisher_name.md>

See the minimal publisher documentation example:

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

The example above specifies the publisher name, a short description, path to
icon to use, and a longer free-form markdown documentation.

### Publisher name guideline

Your publisher name can be your GitHub username or the name of the GitHub
organization you manage.
