# Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities to use Modules with Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import logging
import tensorflow as tf
from tensorflow_hub import tf_utils
from tensorflow_hub import tf_v1


# A collection of pairs (key: string, module: Module) used internally to
# propagate modules from where they are defined to the export hook.
# The collection key is a tuple (not a string) in order to make it invisible
# from user apis such as `get_all_collection_keys()` and manual exporting to
# meta_graphs.
_EXPORT_MODULES_COLLECTION = ("__tfhub_export_modules",)


def register_module_for_export(module, export_name):
  """Register a Module to be exported under `export_name`.

  DEPRECATION NOTE: This belongs to the hub.Module API and TF1 Hub format.

  This function registers `module` to be exported by `LatestModuleExporter`
  under a subdirectory named `export_name`.

  Note that `export_name` must be unique for each module exported from the
  current graph. It only controls the export subdirectory name and it has
  no scope effects such as the `name` parameter during Module instantiation.

  Args:
    module: Module instance to be exported.
    export_name: subdirectory name to use when performing the export.

  Raises:
    ValueError: if `export_name` is already taken in the current graph.
  """
  for used_name, _ in tf_v1.get_collection(_EXPORT_MODULES_COLLECTION):
    if used_name == export_name:
      raise ValueError(
          "There is already a module registered to be exported as %r"
          % export_name)
  tf_v1.add_to_collection(_EXPORT_MODULES_COLLECTION, (export_name, module))


class LatestModuleExporter(tf_v1.estimator.Exporter):
  """Regularly exports registered modules into timestamped directories.

  DEPRECATION NOTE: This belongs to the hub.Module API and TF1 Hub format.

  Modules can be registered to be exported by this class by calling
  `register_module_for_export` when constructing the graph. The
  `export_name` provided determines the subdirectory name used when
  exporting.

  In addition to exporting, this class also garbage collects older exports.

  Example use with EvalSpec:

  ```python
    train_spec = tf.estimator.TrainSpec(...)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        exporters=[
            hub.LatestModuleExporter("tf_hub", serving_input_fn),
        ])
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  ```

  See `LatestModuleExporter.export()` for a direct use example.
  """

  def __init__(self, name, serving_input_fn, exports_to_keep=5):
    """Creates an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter`, which will be used in the export
        path.
      serving_input_fn: A function with no arguments that returns a
        ServingInputReceiver. This is used with the `estimator` passed
        to `export()` to build the graph (in PREDICT mode) that registers the
        modules for export. The model in that graph is never run, so the actual
        data provided by this input fn does not matter.
      exports_to_keep: Number of exports to keep. Older exports will be garbage
        collected. Defaults to 5. Set to None to disable garbage collection.

    Raises:
      ValueError: if any argument is invalid.
    """
    self._name = name
    self._serving_input_fn = serving_input_fn

    self._exports_to_keep = exports_to_keep
    if exports_to_keep is not None and exports_to_keep <= 0:
      raise ValueError(
          "`exports_to_keep`, if provided, must be a positive number")

  @property
  def name(self):
    return self._name

  def export(self, estimator, export_path, checkpoint_path=None,
             eval_result=None, is_the_final_export=None):
    """Actually performs the export of registered Modules.

    This method creates a timestamped directory under `export_path`
    with one sub-directory (named `export_name`) per module registered
    via `register_module_for_export`.

    Example use:

    ```python
      estimator = ... (Create estimator with modules registered for export)...
      exporter = hub.LatestModuleExporter("tf_hub", serving_input_fn)
      exporter.export(estimator, export_path, estimator.latest_checkpoint())
    ```

    Args:
      estimator: the `Estimator` from which to export modules.
      export_path: A string containing a directory where to write the export
        timestamped directories.
      checkpoint_path: The checkpoint path to export. If `None`,
        `estimator.latest_checkpoint()` is used.
      eval_result: Unused.
      is_the_final_export: Unused.

    Returns:
      The path to the created timestamped directory containing the exported
      modules.
    """
    if checkpoint_path is None:
      checkpoint_path = estimator.latest_checkpoint()

    export_dir = tf_utils.get_timestamped_export_dir(export_path)
    temp_export_dir = tf_utils.get_temp_export_dir(export_dir)

    session = _make_estimator_serving_session(estimator, self._serving_input_fn,
                                              checkpoint_path)
    with session:
      export_modules = tf_v1.get_collection(_EXPORT_MODULES_COLLECTION)
      if export_modules:
        for export_name, module in export_modules:
          module_export_path = os.path.join(temp_export_dir,
                                            tf.compat.as_bytes(export_name))
          module.export(module_export_path, session)
        tf_v1.gfile.Rename(temp_export_dir, export_dir)
        tf_utils.garbage_collect_exports(export_path, self._exports_to_keep)
        return export_dir
      else:
        logging.warn("LatestModuleExporter found zero modules to export. "
                     "Use hub.register_module_for_export() if needed.")
        # No export_dir has been created.
        return None


def _make_estimator_serving_session(estimator, serving_input_fn,
                                    checkpoint_path):
  """Returns a session constructed using `estimator` and `serving_input_fn`.

  The Estimator API does not provide an API to construct a graph and session,
  making it necessary for this function to replicate how an estimator builds
  a graph.

  This code is based on `Estimator.export_savedmodel` (another function that
  has to replicate how an estimator builds a graph).

  Args:
    estimator: tf.Estimator to use when constructing the session.
    serving_input_fn: A function that takes no arguments and returns a
      `ServingInputReceiver`. It is used to construct the session.
    checkpoint_path: The checkpoint path to restore in the session. Must not
      be None.
  """
  with tf.Graph().as_default() as g:
    mode = tf_v1.estimator.ModeKeys.PREDICT
    tf_v1.train.create_global_step(g)
    tf_v1.set_random_seed(estimator.config.tf_random_seed)
    serving_input_receiver = serving_input_fn()

    estimator_spec = estimator.model_fn(
        features=serving_input_receiver.features,
        labels=None,
        mode=mode,
        config=estimator.config)

    # pylint: disable=protected-access
    # Note that MonitoredSession(), despite the name is not a Session, and
    # can't be used to export Modules as one can't use them with Savers.
    # As so this must use a raw tf.Session().
    session = tf_v1.Session(config=estimator._session_config)
    # pylint: enable=protected-access

    with session.as_default():
      # TODO(b/71839662): Consider if this needs to support TPUEstimatorSpec
      # which does not have a scaffold member.
      saver_for_restore = estimator_spec.scaffold.saver or tf_v1.train.Saver(
          sharded=True)
      saver_for_restore.restore(session, checkpoint_path)
    return session
