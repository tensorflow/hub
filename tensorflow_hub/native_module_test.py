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
"""Tests for tensorflow_hub.native_module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow_hub import module_def_pb2
from tensorflow_hub import native_module
from tensorflow_hub import tf_v1

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.eager import function as function_eager
from tensorflow.python.framework import function
from tensorflow.python.framework import test_util
from tensorflow.python.ops.control_flow_ops import ControlFlowContext
from tensorflow.python.ops.lookup_ops import HashTable
from tensorflow.python.ops.lookup_ops import index_to_string_table_from_file
from tensorflow.python.ops.lookup_ops import KeyValueTensorInitializer
# pylint: enable=g-direct-tensorflow-import


def load_module_spec(spec):
  """Force use of native_module implementation."""
  return native_module.Loader()(spec)


def multi_signature_module():
  x = tf_v1.placeholder(tf.float32, shape=[None])
  native_module.add_signature("double", {"x": x}, {"y": 2*x})

  z = tf_v1.placeholder(tf.float32, shape=[None])
  native_module.add_signature("square", {"z": z}, {"z_out": z*z})


def batch_norm_module(training):
  x = tf_v1.placeholder(tf.float32, shape=[None, 3])
  y = tf_v1.layers.batch_normalization(x, training=training)
  native_module.add_signature(inputs=x, outputs=y)


def module_with_variables():
  tf_v1.get_variable(
      name="weights",
      shape=[3],
      initializer=tf_v1.zeros_initializer())
  tf_v1.get_variable(
      name="partition",
      shape=[4],
      initializer=tf_v1.zeros_initializer(),
      partitioner=tf_v1.fixed_size_partitioner(3))
  hub.add_signature(outputs=tf.constant(1.0))


class NativeModuleTest(tf.test.TestCase):

  def testModuleWithMissingRequiredFeature(self):
    path = os.path.join(self.get_temp_dir(), "required-feature")
    tf_v1.gfile.MakeDirs(path)
    proto_path = native_module.get_module_proto_path(path)
    with tf_v1.gfile.Open(proto_path, mode="wb") as f:
      module_def_proto = module_def_pb2.ModuleDef()
      module_def_proto.format = module_def_pb2.ModuleDef.FORMAT_V3
      module_def_proto.required_features.extend(["foo-test-missing"])
      f.write(module_def_proto.SerializeToString())
    with self.assertRaisesRegexp(ValueError, "foo-test-missing"):
      load_module_spec(path)

  def testMultiSignatureSpec(self):
    spec = native_module.create_module_spec(multi_signature_module)
    self.assertAllEqual(sorted(spec.get_signature_names()),
                        ["double", "square"])
    self.assertAllEqual(list(spec.get_input_info_dict("double").keys()), ["x"])
    self.assertAllEqual(list(spec.get_output_info_dict("double").keys()), ["y"])
    self.assertAllEqual(list(spec.get_input_info_dict("square").keys()), ["z"])
    self.assertAllEqual(list(spec.get_output_info_dict("square").keys()),
                        ["z_out"])

  def testDefaultTagSpec(self):
    spec = native_module.create_module_spec(multi_signature_module)
    self.assertAllEqual(sorted(spec.get_tags()), [set()])

  def testMultiTagSpec(self):
    spec = native_module.create_module_spec(
        batch_norm_module,
        [({"training"}, {"training": True}),
         ({"inference"}, {"training": False})])
    self.assertAllEqual(sorted(spec.get_tags()),
                        [set(["training"]), set(["inference"])])

  def testModuleWithVariablesAndNoCheckpoint(self):
    with tf.Graph().as_default():
      spec = native_module.create_module_spec(module_with_variables)
      spec._create_impl(name="module", trainable=False, tags=None)
      self.assertAllEqual(
          [x.op.name for x in tf_v1.global_variables()],
          [
              "module/weights",
              "module/partition/part_0",
              "module/partition/part_1",
              "module/partition/part_2",
          ])

      with tf_v1.Session() as session:
        session.run(tf_v1.initializers.global_variables())
        expected_values = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0],
            [0.0],
            [0.0],
        ]
        for a, b in zip(session.run(tf_v1.global_variables()), expected_values):
          self.assertAllEqual(a, b)

  def testNoSignaturesPresent(self):

    def wrong_module_fn():
      x = tf_v1.placeholder(tf.float32, shape=[None, 3])
      return tf.identity(x)

    with self.assertRaises(ValueError) as cm:
      spec = native_module.create_module_spec(wrong_module_fn)
    self.assertIn("No signatures present", str(cm.exception))

  def testUnsupportedCollections(self):

    def module_fn():
      scale = tf_v1.get_variable("x", (), collections=["my_scope"])
      x = tf_v1.placeholder(tf.float32, shape=[None, 3])
      native_module.add_signature("my_func", {"x": x}, {"y": x*scale})

    with self.assertRaises(ValueError) as cm:
      _ = native_module.create_module_spec(module_fn)
      self.assertIn("Unsupported collections in graph", cm)

    with tf.Graph().as_default() as tmp_graph:
      module_fn()
      unsupported_collections = native_module.get_unsupported_collections(
          tmp_graph.get_all_collection_keys())
      self.assertEqual(["my_scope"], unsupported_collections)

    _ = native_module.create_module_spec(
        module_fn, drop_collections=unsupported_collections)


class RecoverPartitionedVariableMapTest(tf.test.TestCase):

  def testRecoverPartitionedVariableMap(self):
    with tf.Graph().as_default():
      with tf_v1.variable_scope("test"):
        partitioner = tf_v1.fixed_size_partitioner(3)
        tf_v1.get_variable(
            initializer=tf.ones([11, 5]),
            name="partitioned_variable",
            partitioner=partitioner)
        tf_v1.get_variable(
            initializer=tf.ones([11, 5]),
            name="normal_variable")

      all_vars = tf_v1.global_variables()
      all_vars_dict = {var.op.name[5:]: var for var in all_vars}
      self.assertEqual(set(all_vars_dict.keys()), set([
          "partitioned_variable/part_0",
          "partitioned_variable/part_1",
          "partitioned_variable/part_2",
          "normal_variable"]))

      self.assertEqual(len(all_vars_dict), 4)
      var_map = native_module.recover_partitioned_variable_map(all_vars_dict)
      self.assertEqual(set(var_map.keys()), set([
          "partitioned_variable", "normal_variable"]))

      # Verify order of the partitioned variable list
      self.assertAllEqual(
          [v.op.name for v in var_map["partitioned_variable"]],
          [
              "test/partitioned_variable/part_0",
              "test/partitioned_variable/part_1",
              "test/partitioned_variable/part_2",
          ])


def stateless_module_fn():
  x = tf_v1.placeholder(tf.int64)
  y = x*x
  hub.add_signature(inputs=x, outputs=y)


def unused_input_module_fn():
  x = tf_v1.placeholder(tf.int64)
  y = tf_v1.placeholder(tf.int64)
  result = x*x
  hub.add_signature(
      inputs={"x": x, "unused": y},
      outputs=result)


def double_module_fn():
  w = tf.Variable(2.0)
  x = tf_v1.placeholder(dtype=tf.float32)
  hub.add_signature(inputs=x, outputs=x*w)


def create_partitioned_variable_module_fn(partitions, shape):
  """Returns a module summing one normal and one partitioned variable."""
  def module_fn():
    """A module summing one normal and one partitioned variable."""
    partitioner = tf_v1.fixed_size_partitioner(partitions)
    var_1 = tf_v1.get_variable(
        initializer=tf.ones(shape),
        name="partitioned_variable",
        partitioner=partitioner)
    var_2 = tf_v1.get_variable(
        initializer=tf.ones(shape), name="normal_variable")
    hub.add_signature(outputs=var_1 + var_2)

  return module_fn


class TFHubStatelessModuleTest(tf.test.TestCase):

  def testLoadModuleFromFuncDef(self):
    with tf_v1.Session() as sess:
      v = tf_v1.placeholder(tf.int64)
      spec = hub.create_module_spec(stateless_module_fn)
      m = hub.Module(spec)
      y = m(v)
      self.assertEqual(sess.run(y, feed_dict={v: 10}), 100)

  def testUnusedInputModule(self):
    with tf_v1.Session() as sess:
      v1 = tf_v1.placeholder(tf.int64)
      v2 = tf_v1.placeholder(tf.int64)
      spec = hub.create_module_spec(unused_input_module_fn)
      m = hub.Module(spec)
      out = m({"x": v1, "unused": v2})
      self.assertEqual(sess.run(out, feed_dict={v1: 10, v2: 4}), 100)

  def testConvertToTensor(self):
    spec = hub.create_module_spec(stateless_module_fn)
    with tf_v1.Session() as sess:
      m = hub.Module(spec)
      y = m([10, 2])
      self.assertAllEqual(sess.run(y), [100, 4])
    with tf_v1.Session() as sess:
      m = hub.Module(spec)
      with self.assertRaises(TypeError):
        m("hello")

  def testArgErrors(self):
    spec = hub.create_module_spec(stateless_module_fn)
    with tf_v1.Session():
      m = hub.Module(spec)
      with self.assertRaisesRegexp(TypeError, "missing"):
        m()

  @test_util.run_v1_only("b/138681007")
  def testUseWithinWhileLoop(self):
    with tf.Graph().as_default():
      spec = hub.create_module_spec(double_module_fn)
      m = hub.Module(spec)
      i = tf.constant(0)
      x = tf.constant(10.0)
      p = tf_v1.placeholder(dtype=tf.int32)
      c = lambda i, x: tf.less(i, p)
      b = lambda i, x: (tf.add(i, 1), m(x))
      oi, ox = tf.while_loop(c, b, [i, x])
      dox = tf.gradients(ox, x)[0]
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllEqual(sess.run([oi, ox], feed_dict={p: 1}), [1, 20])
        self.assertAllEqual(sess.run([oi, ox], feed_dict={p: 2}), [2, 40])
        self.assertAllEqual(sess.run([oi, ox], feed_dict={p: 4}), [4, 160])
        # Gradients also use the control flow structures setup earlier.
        # Also check they are working properly.
        self.assertAllEqual(sess.run([dox], feed_dict={p: 1}), [2])
        self.assertAllEqual(sess.run([dox], feed_dict={p: 2}), [4])
        self.assertAllEqual(sess.run([dox], feed_dict={p: 4}), [16])

  def testClearControlDependenciesForModuleStateButNotApplyGraphs(self):
    module_spec = hub.create_module_spec(stateless_module_fn)

    with tf.Graph().as_default() as g1:
      v = tf_v1.placeholder(dtype=tf.int64, name="v")
      m = hub.Module(module_spec)
      m(v)

    with tf.Graph().as_default() as g2:
      v = tf_v1.placeholder(dtype=tf.int64, name="v")
      with tf.control_dependencies([v]):
        m = hub.Module(module_spec)
      m(v)

    self.assertEqual(g1.as_graph_def(), g2.as_graph_def())

    with tf.Graph().as_default() as g3:
      v = tf_v1.placeholder(dtype=tf.int64, name="v")
      m = hub.Module(module_spec)
      m(v)

    with tf.Graph().as_default() as g4:
      v = tf_v1.placeholder(dtype=tf.int64, name="v")
      m = hub.Module(module_spec)
      with tf.control_dependencies([v]):
        m(v)

    self.assertNotEqual(g3.as_graph_def(), g4.as_graph_def())


def sparse_square_module_fn():
  x = tf_v1.sparse_placeholder(dtype=tf.int64, name="x")
  out = tf.SparseTensor(x.indices, x.values * x.values, x.dense_shape)
  hub.add_signature(inputs=x, outputs=out)


class TFHubSparseTensorModuleTest(tf.test.TestCase):

  def testSparseTensors(self):
    square_spec = hub.create_module_spec(sparse_square_module_fn)

    with tf.Graph().as_default():
      square = hub.Module(square_spec)
      v = tf_v1.sparse_placeholder(dtype=tf.int64, name="v")
      y = square(v)

      with tf_v1.Session().as_default():
        indices = [[0, 0], [0, 1], [1, 1]]
        values = [10, 2, 1]
        shape = [2, 2]
        v1 = tf_v1.SparseTensorValue(indices, values, shape)
        v2 = y.eval(feed_dict={v: v1})
        v4 = y.eval(feed_dict={v: v2})

        self.assertAllEqual(v4.indices, indices)  # Unchanged.
        self.assertAllEqual(v4.values, [t**4 for t in values])  # Squared twice.
        self.assertAllEqual(v4.dense_shape, shape)  # Unchanged.


def stateful_module_fn():
  v = tf_v1.get_variable(
      "var123", shape=[3],
      initializer=tf_v1.constant_initializer([1.0, 2.0, 3.0]))
  hub.add_signature(outputs=v.value())


def stateful_rv_module_fn():
  r = tf_v1.get_variable(
      "rv_var123", shape=[],
      initializer=tf_v1.constant_initializer(10.0),
      use_resource=True)
  hub.add_signature(outputs=r.value())


class TPUReplicateContext(ControlFlowContext):

  def __init__(self):
    super(TPUReplicateContext, self).__init__()
    self._name = "TPUReplicateContext"

  def AddOp(self, _):
    pass

  def AddValue(self, x):
    return x

  def to_control_flow_context_def(self, context_def, export_scope=None):
    super(TPUReplicateContext, self).to_control_flow_context_def(
        context_def, export_scope)


def stateful_random_rv_module_fn():
  r = tf_v1.get_variable(
      "rv_var123",
      shape=[],
      initializer=tf_v1.random_uniform_initializer(),
      use_resource=True)
  hub.add_signature(outputs=r.value())


def stateful_rv_with_input_module_fn():
  x = tf_v1.placeholder(dtype=tf.float32, name="x")
  # Add a placeholder/variable that doesn't go to an output.
  y = tf_v1.placeholder(dtype=tf.float32, name="y")
  r = tf_v1.get_variable(
      "rv_var123",
      shape=[],
      initializer=tf_v1.constant_initializer(10.0),
      use_resource=True)
  t = tf_v1.get_variable(
      "rv_var456",
      shape=[],
      initializer=tf_v1.constant_initializer(10.0),
      use_resource=True)
  t.assign(y)
  res = x + r
  hub.add_signature(inputs={"x": x}, outputs=res)


def control_dependency_module_fn():
  const_op = tf.constant(1.0, name="dependency_op")
  with tf.control_dependencies([const_op]):
    res = tf.constant(3.0) + tf.constant(2.0)
  hub.add_signature(inputs={}, outputs=res)


def stateful_non_rv_module_fn():
  v = tf_v1.get_variable(
      "var123", shape=[],
      initializer=tf_v1.constant_initializer(10.0),
      use_resource=False)
  hub.add_signature(outputs=v.value())


def stateful_module_fn_with_colocation():
  v = tf_v1.get_variable(
      "var123", shape=[],
      initializer=tf_v1.constant_initializer(1.0),
      use_resource=False)
  v_value = v.value()
  x = tf_v1.placeholder(dtype=tf.float32, name="x")
  with tf_v1.colocate_with(v), tf_v1.colocate_with(x):
    y = tf.add(v_value, x, name="y")
  hub.add_signature(inputs=x, outputs=y)


class TFHubStatefulModuleTest(tf.test.TestCase):

  def testVariables(self):
    with tf.Graph().as_default():
      spec = hub.create_module_spec(stateful_module_fn)
      m = hub.Module(spec, name="test")
      out = m()
      self.assertEqual(list(m.variable_map.keys()), ["var123"])
      self.assertEqual(m.variable_map["var123"].name, "test/var123:0")
      self.assertEqual([v.name for v in m.variables], ["test/var123:0"])
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllClose(sess.run(out), [1.0, 2.0, 3.0])

  def testResourceVariables(self):
    with tf.Graph().as_default():
      spec = hub.create_module_spec(stateful_rv_module_fn)
      m = hub.Module(spec, name="test_rv")
      out = m()
      self.assertEqual(list(m.variable_map.keys()), ["rv_var123"])
      self.assertEqual(m.variable_map["rv_var123"].name, "test_rv/rv_var123:0")
      self.assertEqual([v.name for v in m.variables], ["test_rv/rv_var123:0"])

      # Check that "shared_name" attributes are adapted correctly:
      var_handle_op_name = "test_rv/rv_var123"
      var_handle_op = tf_v1.get_default_graph().get_operation_by_name(
          var_handle_op_name)
      self.assertEqual(
          var_handle_op.get_attr("shared_name"),
          tf.compat.as_bytes(var_handle_op_name))

      export_path = os.path.join(self.get_temp_dir(), "resource-variables")
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllClose(sess.run(out), 10.0)
        m.export(export_path, sess)

    with tf.Graph().as_default():
      f = hub.Module(export_path)
      out = f()

      # Test colocation constraints on the read op in the apply graph.
      # It has two legal values:
      # - Colocation with the VarHandleOp in the state graph.
      # - No constraint, in which case it reports its own colocation_group.
      #   This appears to happen at the time of this writing (March 2018)
      #   because the Python code relies on the TensorFlow core to handle
      #   VariableReadOps as a special case and colocate them with their
      #   VarHandleOp input, which is mapped to the state graph.
      # In any case, the point is to *not* colocate with the stillborn copy
      # of the VarHandleOp in the apply graph scope.
      if out.op.colocation_groups() != [
          tf.compat.as_bytes("loc:@" + out.op.name)]:
        self.assertItemsEqual(out.op.colocation_groups(),
                              [tf.compat.as_bytes("loc:@module/rv_var123")])

      # Check that "shared_name" attributes are adapted correctly:
      var_handle_op_name = "module/rv_var123"
      var_handle_op = tf_v1.get_default_graph().get_operation_by_name(
          var_handle_op_name)
      self.assertEqual(
          var_handle_op.get_attr("shared_name"),
          tf.compat.as_bytes(var_handle_op_name))

      # Create a saver for the whole graph.
      saver = tf_v1.train.Saver()

      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllClose(sess.run(out), 10.0)

        # Make sure that the variable names stored in a checkpoint of the graph
        # are as expected.
        variables_path = os.path.join(self.get_temp_dir(), "variables")
        saver.save(
            sess, variables_path, write_meta_graph=False, write_state=False)
        variable_names_and_shapes = tf_v1.train.list_variables(
            ckpt_dir_or_file=variables_path)
        variable_names = set(name for name, _ in variable_names_and_shapes)
        self.assertEqual(variable_names, {"module/rv_var123"})

  def testNonResourceVariables(self):
    with tf.Graph().as_default():
      spec = hub.create_module_spec(stateful_non_rv_module_fn)
      m = hub.Module(spec, name="test_non_rv")
      out = m()
      self.assertEqual(list(m.variable_map.keys()), ["var123"])
      self.assertEqual(m.variable_map["var123"].name, "test_non_rv/var123:0")
      self.assertEqual([v.name for v in m.variables], ["test_non_rv/var123:0"])

      export_path = os.path.join(self.get_temp_dir(), "non-resource-variables")
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllClose(sess.run(out), 10.0)
        m.export(export_path, sess)

      with tf.Graph().as_default():
        f = hub.Module(export_path)
        out = f()

        # Test that the read op in the apply graph gets colocated with the
        # variable in the state graph scope "module/" (and not the stillborn
        # copy in the apply graph scope).
        self.assertItemsEqual(out.op.colocation_groups(),
                              [tf.compat.as_bytes("loc:@module/var123")])

        # Create a saver for the whole graph.
        saver = tf_v1.train.Saver()

        with tf_v1.Session() as sess:
          sess.run(tf_v1.global_variables_initializer())
          self.assertAllClose(sess.run(out), 10.0)

          # Make sure that the variable names stored in a checkpoint of the
          # graph are as expected.
          variables_path = os.path.join(self.get_temp_dir(), "variables")
          saver.save(
              sess, variables_path, write_meta_graph=False, write_state=False)
          variable_names_and_shapes = tf_v1.train.list_variables(
              ckpt_dir_or_file=variables_path)
          variable_names = set(name for name, _ in variable_names_and_shapes)
          self.assertEqual(variable_names, {"module/var123"})

  @test_util.run_v1_only("b/138681007")
  def testNonResourceVariableInWhileLoop(self):
    with tf.Graph().as_default():
      # This test uses non-Resource variables to see an actual colocation
      # constraint propagated to the context Enter op. The long comment on
      # colocation in testResourceVariables explains why they may not offer
      # that.
      spec = hub.create_module_spec(stateful_non_rv_module_fn)
      m = hub.Module(spec)
      cond = lambda i, x: tf.less(i, 4)
      def body(i, x):
        v = m()
        self.assertItemsEqual(v.op.colocation_groups(),
                              [tf.compat.as_bytes("loc:@module/var123")])
        return (tf.add(i, 1), 2*x)
      oi, ox = tf.while_loop(cond, body, [0, 10.0])
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllEqual(sess.run([oi, ox]), [4, 160.0])

  @test_util.run_v1_only("b/138681007")
  def testNonResourceVariableInCond(self):
    with tf.Graph().as_default():
      spec = hub.create_module_spec(stateful_non_rv_module_fn)
      m = hub.Module(spec)
      pred = tf_v1.placeholder(tf.bool)
      def true_fn():
        v = m()
        self.assertItemsEqual(v.op.colocation_groups(),
                              [tf.compat.as_bytes("loc:@module/var123")])
        return v
      def false_fn():
        return tf.constant(9.0)
      out = tf.cond(pred, true_fn, false_fn)
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertEqual(sess.run(out, feed_dict={pred: True}), 10.0)
        self.assertEqual(sess.run(out, feed_dict={pred: False}), 9.0)

  def testVariableColocationPropagation(self):
    with tf.Graph().as_default():
      spec = hub.create_module_spec(stateful_module_fn_with_colocation)
      m = hub.Module(spec)
      u1 = tf.constant(1, name="u1")
      u2 = tf.constant(2, name="u2")
      with tf_v1.colocate_with(u1), tf_v1.colocate_with(u2):
        x = tf.constant(100.0, name="x")
      y = m(x)
      self.assertItemsEqual(y.op.colocation_groups(),
                            [tf.compat.as_bytes("loc:@module/var123"),
                             tf.compat.as_bytes("loc:@u1"),
                             tf.compat.as_bytes("loc:@u2")])
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertEqual(sess.run(y), 101.0)

  def testPartitionedVariables(self):
    with tf.Graph().as_default():
      spec = hub.create_module_spec(
          create_partitioned_variable_module_fn(partitions=3, shape=[7, 3]))
      m = hub.Module(spec, name="test")
      out = m()
      self.assertEqual(len(m.variable_map), 2)
      self.assertEqual(m.variable_map["normal_variable"].name,
                       "test/normal_variable:0")
      self.assertAllEqual([
          variable.name for variable in m.variable_map["partitioned_variable"]
      ], [
          "test/partitioned_variable/part_0:0",
          "test/partitioned_variable/part_1:0",
          "test/partitioned_variable/part_2:0"
      ])
      self.assertAllEqual(  # Check deterministric order (by variable_map key).
          [variable.name for variable in m.variables],
          ["test/normal_variable:0",
           "test/partitioned_variable/part_0:0",
           "test/partitioned_variable/part_1:0",
           "test/partitioned_variable/part_2:0"])
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllClose(sess.run(out), 2 * np.ones([7, 3]))

  def testLargePartitionedVariables(self):
    with tf.Graph().as_default():
      spec = hub.create_module_spec(
          create_partitioned_variable_module_fn(partitions=25, shape=[600, 3]))
      m = hub.Module(spec, name="test")
      out = m()
      self.assertEqual(len(m.variable_map), 2)
      self.assertEqual(len(m.variable_map["partitioned_variable"]), 25)
      self.assertEqual(len(m.variables), 26)
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllClose(sess.run(out), 2 * np.ones([600, 3]))

  def testLoadTrainableModuleFromFuncDef(self):
    with tf_v1.Session() as sess:
      spec = hub.create_module_spec(stateful_module_fn)
      m = hub.Module(spec, trainable=True)
      x = m()
      step = tf.Variable(0, trainable=False, name="global_step")
      train_op = tf_v1.train.GradientDescentOptimizer(0.40).minimize(
          loss=tf_v1.losses.mean_squared_error(x, [3.1, 3.2, 3.3]),
          global_step=step)
      sess.run(tf_v1.global_variables_initializer())
      for _ in range(50):
        sess.run(train_op)
      got = sess.run(x)
      self.assertAllClose(got, [3.1, 3.2, 3.3])

  # TODO(b/112575006): The following tests verify functionality of function call
  # within a TPU context. Work to generalize this for all function calls is
  # ongoing.
  def testTPUModuleInitializeOnceWithDefun(self):
    spec = hub.create_module_spec(stateful_random_rv_module_fn)

    @function.Defun()
    def import_computation():
      context = TPUReplicateContext()
      context.Enter()
      m = hub.Module(spec, name="module_", trainable=True)
      return [m(), m()]

    with tf_v1.Graph().as_default(), tf_v1.Session() as sess:
      x = import_computation()
      sess.run(tf_v1.global_variables_initializer())
      got = sess.run(x)
      # Check the values are equal. If the initializer ran on each call,
      # the values would be different.
      self.assertEqual(got[0], got[1])

  def testTPUPruneWithUnusedInput(self):
    spec = hub.create_module_spec(unused_input_module_fn)

    @function.Defun()
    def import_computation(x):
      context = TPUReplicateContext()
      context.Enter()
      m = hub.Module(spec, name="module_", trainable=True)
      return m({
          "x": tf.cast(x, dtype=tf.int64),
          "unused": tf.constant(2, dtype=tf.int64)
      })

    with tf_v1.Graph().as_default(), tf_v1.Session() as sess:
      x = import_computation(5)
      got = sess.run(x)
      self.assertEqual(got, 25)

  def testTPUModuleDoesntPruneControlDependencies(self):
    spec = hub.create_module_spec(control_dependency_module_fn)

    @function.Defun()
    def import_computation():
      context = TPUReplicateContext()
      context.Enter()
      m = hub.Module(spec, name="module_", trainable=True)
      return m()

    with tf_v1.Graph().as_default(), tf_v1.Session() as sess:
      x = import_computation()
      got = sess.run(x)
      self.assertEqual(got, 5.0)
      # If the op got pruned, the following get_operation_by_name should fail
      # with a dependency error.
      tf_v1.get_default_graph().get_operation_by_name("module_/dependency_op")

  def testTPUModuleWithDefun(self):
    spec = hub.create_module_spec(stateful_rv_with_input_module_fn)

    @function.Defun()
    def import_computation(first, second):
      context = TPUReplicateContext()
      context.Enter()
      m = hub.Module(spec, name="module_", trainable=True)
      return [m(first), m(second)]

    with tf_v1.Graph().as_default(), tf_v1.Session() as sess:
      x = import_computation(9.0, 6.0)
      sess.run(tf_v1.global_variables_initializer())
      got = sess.run(x)
      self.assertEqual(got, (19.0, 16.0))

  def testTPUModuleWithTFEDefun(self):
    with tf_v1.Graph().as_default() as graph, tf_v1.Session() as sess:
      spec = hub.create_module_spec(stateful_rv_with_input_module_fn)

      @function_eager.defun()
      def import_computation(first, second):
        context = TPUReplicateContext()
        context.Enter()
        m = hub.Module(spec, trainable=True)
        return [m(first), m(second)]

      x = import_computation(9.0, 6.0)
      sess.run(tf_v1.global_variables_initializer())
      got = sess.run(x)
      self.assertEqual(got, [19.0, 16.0])

  def testTPUModuleWithWrapFunc(self):
    spec = hub.create_module_spec(stateful_rv_with_input_module_fn)

    def import_computation(first, second):
      context = TPUReplicateContext()
      context.Enter()
      m = hub.Module(spec, trainable=True)
      return [m(first), m(second)]

    with tf_v1.Graph().as_default(), tf_v1.Session() as sess:
      x = tf_v1.wrap_function(
          import_computation,
          [tf.TensorSpec((), tf.float32),
           tf.TensorSpec((), tf.float32)])
      sess.run(tf_v1.global_variables_initializer())
      got = sess.run(x(9.0, 6.0))
      self.assertEqual(got, [19.0, 16.0])

  def _exportModulewithTrainedVariable(self):
    export_path = os.path.join(self.get_temp_dir(), "var-module")
    with tf.Graph().as_default():
      spec = hub.create_module_spec(stateful_module_fn)
      m = hub.Module(spec, trainable=True)
      assign_op = tf_v1.assign(m.variable_map["var123"],
                               tf.constant([9.0, 9.0, 9.0]))
      with tf_v1.Session() as sess:
        sess.run(assign_op)
        m.export(export_path, sess)
    return export_path

  def testModuleWithTrainedVariable(self):
    with tf.Graph().as_default():
      f = hub.Module(self._exportModulewithTrainedVariable())
      out = f()
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        got = sess.run(out)
        self.assertAllClose(got, [9.0, 9.0, 9.0])

  def testModuleEvalWithTrainedVariable(self):
    export_path = self._exportModulewithTrainedVariable()
    with hub.eval_function_for_module(export_path) as f:
      self.assertAllClose(f(), [9.0, 9.0, 9.0])


def table_lookup_module_fn():
  x = tf_v1.placeholder(dtype=tf.int64, name="x")
  keys = tf.constant([0, 1, 2], dtype=tf.int64)
  values = tf.constant(["index0", "hello", "world"])

  tbl_init = KeyValueTensorInitializer(keys, values)
  table = HashTable(tbl_init, "UNK")
  hub.add_signature(inputs=x, outputs=table.lookup(x))


class TFHubTableLookupModuleTest(tf.test.TestCase):

  def _exportModuleWithTable(self):
    export_path = os.path.join(self.get_temp_dir(), "table-module")
    with tf.Graph().as_default():
      spec = hub.create_module_spec(table_lookup_module_fn)
      m = hub.Module(spec)
      # Export requires a session to work regardless of the module having no
      # variables to export.
      with tf_v1.Session() as sess:
        m.export(export_path, sess)
    return export_path

  def testModuleWithTable(self):
    with tf.Graph().as_default():
      v = tf_v1.placeholder(dtype=tf.int64)
      f = hub.Module(self._exportModuleWithTable())
      y = f(v)
      with tf_v1.Session() as sess:
        sess.run(tf_v1.tables_initializer())
        got = sess.run(y, feed_dict={v: [0, 1, 2, 3]})
        self.assertAllEqual(list(got), [b"index0", b"hello", b"world", b"UNK"])

  def testModuleEvalWithTable(self):
    with hub.eval_function_for_module(self._exportModuleWithTable()) as f:
      got = f([0, 1, 2, 3])
      self.assertAllEqual(list(got), [b"index0", b"hello", b"world", b"UNK"])


def do_table_lookup(indices, vocabulary_file):
  table = index_to_string_table_from_file(
      vocabulary_file=vocabulary_file,
      default_value="UNKNOWN")
  return table.lookup(indices)


def layers_module_fn():
  """Module that exercises the use of layers."""
  # This is a plain linear map Mx+b regularized by the sum of the squares
  # of the coefficients in M and b.
  x = tf_v1.placeholder(dtype=tf.float32, shape=[None, 2], name="x")
  def l2(weights):
    """Applies l2 regularization to weights."""
    with tf.control_dependencies([weights]):
      return 2.0 * tf_v1.nn.l2_loss(weights)

  h = tf_v1.layers.dense(
      x, 2,
      activation=None,
      kernel_regularizer=l2,
      bias_regularizer=l2)
  hub.add_signature(inputs=x, outputs=h)


class TFHubLayersModuleTest(tf.test.TestCase):

  def testModuleWithLayers(self):
    export_path = os.path.join(self.get_temp_dir(), "layers-module")

    sample_input = [[1.0, 2.0], [3.1, 10.0]]

    spec = hub.create_module_spec(layers_module_fn)
    with tf.Graph().as_default():
      m = hub.Module(spec, trainable=False)
      x = tf_v1.placeholder(dtype=tf.float32)
      y = m(x)
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        sample_output = sess.run(y, feed_dict={x: sample_input})
        m.export(export_path, sess)

    with tf.Graph().as_default():
      x = tf_v1.placeholder(dtype=tf.float32)
      y = hub.Module(export_path)(x)

      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        got = sess.run(y, feed_dict={x: sample_input})
        self.assertAllEqual(got, sample_output)

  def testModuleWithRegularizedLayers(self):
    # The linear map y = Mx + b with L2 regularization on M and b
    # when trained at x = [1,1] with L2 loss towards the target y' = [4,4]
    # learns M = [[1,1],[1,1]], b = [1,1], y = [3,3], with eight balanced
    # loss terms: the elements of M, b, and y' - y are all distance 1 from zero.
    train_input = [[1.0, 1.0]]
    target = [[4.0, 4.0]]

    spec = hub.create_module_spec(layers_module_fn)
    with tf.Graph().as_default():
      m = hub.Module(spec, trainable=True)
      x = tf_v1.placeholder(dtype=tf.float32)
      y = m(x)
      squared_loss = tf_v1.losses.mean_squared_error(y, target, weights=2.0)
      # Recover REGULARIZATION_LOSSES from the module.
      total_loss = squared_loss + tf_v1.losses.get_regularization_loss()
      step = tf.Variable(0, trainable=False, name="global_step")
      train = tf_v1.train.GradientDescentOptimizer(0.1).minimize(
          loss=total_loss, global_step=step)
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        for _ in range(50):
          sess.run(train, feed_dict={x: train_input})
        # Verify M = [[1,1],[1,1]], b = [1,1] by evaluating at three points.
        # Without regularization, the result would be an underdetermined mess.
        out = sess.run(y, feed_dict={x: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]})
        self.assertAllClose(
            out, [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]], atol=0.001)


def valid_colocation_module_fn():
  w = tf.Variable(42 + 69, name="w")
  # w.op has the same name on resource and non-resource variables
  with tf_v1.colocate_with(w.op):
    # A colocation reference among state nodes is ok.
    v = tf.Variable(1.0, name="v")
    assert v.op.colocation_groups() == [tf.compat.as_bytes("loc:@w")]
    # A colocation reference from other nodes to state nodes is ok.
    y = tf.add(v, 1, name="y")
    assert y.op.colocation_groups() == [tf.compat.as_bytes("loc:@w")]
  x = tf_v1.placeholder(dtype=tf.float32, name="x")
  with tf_v1.colocate_with(x):
    # A colocation reference from other nodes to input nodes is ok.
    z = tf.add(x, 1, name="z")
    assert z.op.colocation_groups() == [tf.compat.as_bytes("loc:@x")]
  hub.add_signature(inputs=dict(x=x), outputs=dict(y=y, z=z))


def bad_input_colocation_module_fn():
  u = tf.add(42, 69, name="u")
  with tf_v1.colocate_with(u):
    # Inputs must not reference other nodes for colocation.
    x = tf_v1.placeholder(tf.float32, name="x")
  y = x + 1.0
  hub.add_signature(inputs=x, outputs=y)


def bad_state_colocation_module_fn():
  u = tf.add(42, 69, name="u")
  with tf_v1.colocate_with(u):
    # State-holding nodes must not reference other nodes for colocation.
    v = tf.Variable(1.0, name="v")
  x = tf_v1.placeholder(dtype=tf.float32)
  y = x + v
  hub.add_signature(inputs=x, outputs=y)


def brittle_multivalued_colocation_module_fn():
  x, y = tf.split([1, 2], 2, name="split")
  with tf_v1.colocate_with(x), tf_v1.colocate_with(y):
    z = tf.add(x, y, name="add")
    assert z.op.colocation_groups() == [tf.compat.as_bytes("loc:@split")]
  hub.add_signature(inputs=dict(x=x, y=y), outputs=z, name="both")
  hub.add_signature(inputs=dict(x=x), outputs=z, name="partial")


class ColocationRewritingTest(tf.test.TestCase):

  def testValidCase(self):
    """Tests a complex, valid case end-to-end."""
    spec = hub.create_module_spec(valid_colocation_module_fn)
    with tf.Graph().as_default():
      u = tf.constant(7.0, name="u")
      m = hub.Module(spec, name="m")
      outputs = m(dict(x=u), as_dict=True)
      self.assertItemsEqual(outputs["y"].op.colocation_groups(),
                            [tf.compat.as_bytes("loc:@m/w")])
      self.assertItemsEqual(outputs["z"].op.colocation_groups(),
                            [tf.compat.as_bytes("loc:@u")])

  def testBadInputColocation(self):
    """Tests catching bad colocation of inputs during create_module_spec."""
    with self.assertRaisesRegexp(ValueError, "(?s)input.*colocate.*loc:@u"):
      _ = hub.create_module_spec(bad_input_colocation_module_fn)

  def testBadStateColocation(self):
    """Tests catching bad colocation of states during create_module_spec."""
    with self.assertRaisesRegexp(ValueError, "(?s)state.*colocate.*loc:@u"):
      _ = hub.create_module_spec(bad_state_colocation_module_fn)

  def testInputsFromMultivaluedOp(self):
    """Tests warning for inputs from multivalued ops in create_module_spec."""
    # Ideally, one would be able to write
    #    with self.assertLogs("blah"): hub.create_module_spec(module_fn)
    # but in the absence of assertions on logs, we test the underlying helper
    # in the environment seen from within a module_fn.
    with tf.Graph().as_default():
      first, _ = tf.split([[1, 2], [3, 4]], 2, name="split1")
      _, second = tf.split([[5, 6], [7, 8]], 2, name="split2")
      third = tf.constant(105, name="const")
      message = native_module.find_signature_inputs_from_multivalued_ops(
          dict(first=first, second=second, third=third))
    self.assertRegexpMatches(
        message,
        ".*single output.*\n"
        "Affected inputs: first='split1:0', second='split2:1'$")
    # Also test the case of no errors.
    with tf.Graph().as_default():
      first = tf.constant(101)
      second = tf.constant(102)
      third = tf.constant(103)
      message = native_module.find_signature_inputs_from_multivalued_ops(
          dict(first=first, second=second, third=third))
    self.assertIsNone(message)

  def testSparseInputsFromMultivaluedOp(self):
    """Tests warning for SparseTensor inputs from multivalued ops."""
    with tf.Graph().as_default():
      one, _ = tf_v1.sparse_split(
          sp_input=tf.SparseTensor(indices=[[0, 1], [1, 2]], values=[1, 2],
                                   dense_shape=[2, 3]),
          num_split=2, axis=0, name="op1")
      _, two = tf_v1.sparse_split(
          sp_input=tf.SparseTensor(indices=[[0, 0], [1, 1]], values=[3, 4],
                                   dense_shape=[2, 3]),
          num_split=2, axis=0, name="op2")
      three = tf.SparseTensor(indices=[[0]], values=[5], dense_shape=[2])
      message = native_module.find_signature_inputs_from_multivalued_ops(
          dict(one=one, two=two, three=three))
    self.assertRegexpMatches(
        message,
        ".*single output.*\nAffected inputs: "
        "one.indices='op1:0', one.values='op1:2', one.dense_shape='op1:4', "
        "two.indices='op2:1', two.values='op2:3', two.dense_shape='op2:5'$")
    # Also test the case of no errors.
    with tf.Graph().as_default():
      one = tf.SparseTensor(indices=[[0]], values=[1], dense_shape=[2])
      two = tf.SparseTensor(indices=[[1]], values=[2], dense_shape=[2])
      message = native_module.find_signature_inputs_from_multivalued_ops(
          dict(one=one, two=two, three=three))
    self.assertIsNone(message)

  def testBrittleColocationWithInputsFromMultivaluedOp(self):
    """Tests handling of ambiguous rewrites during module.__call__."""
    spec = hub.create_module_spec(brittle_multivalued_colocation_module_fn)
    with tf.Graph().as_default():
      u = tf.constant([1], name="u")
      with tf_v1.colocate_with(u):
        v = tf.constant([2], name="v")
      w = tf.constant([3], name="w")
      m = hub.Module(spec, name="m")
      # It works if both inputs are mapped to ops with equal colocation groups.
      assert u.op.colocation_groups() == v.op.colocation_groups()
      z = m(dict(x=u, y=v), signature="both")
      self.assertItemsEqual(z.op.colocation_groups(),
                            [tf.compat.as_bytes("loc:@u")])
      # It crashes in the general case.
      assert u.op.colocation_groups() != w.op.colocation_groups()
      with self.assertRaisesRegexp(
          ValueError,
          # In Python 3 (but not 2), colocation groups are lists of bytes,
          # which are formatted with a leading "b" just before the quotes.
          r"(?s)Failed to rewrite .*b?'loc:@m_apply_both_1/split' .*"
          "\[b?'loc:@[uw]'\] vs \[b?'loc:@[wu]'\]"):
        z = m(dict(x=u, y=w), signature="both")

  def testBadColocationWithPartialInputsFromMultivaluedOp(self):
    spec = hub.create_module_spec(brittle_multivalued_colocation_module_fn)
    with tf.Graph().as_default():
      u = tf.constant([1], name="u")
      m = hub.Module(spec, name="m")
      with self.assertRaisesRegexp(
          ValueError,
          r"(?s)Failed to rewrite .*b?'loc:@m_apply_partial/split' .*"
          "\[b?'loc:@u'\] vs \[b?'loc:@m_apply_partial/split'\]"):
        z = m(dict(x=u), signature="partial")


def update_ops_module_fn():
  counter = tf.Variable(0, trainable=False)
  tf_v1.add_to_collection(tf_v1.GraphKeys.UPDATE_OPS, counter.assign_add(1))
  hub.add_signature(inputs=None, outputs=counter.value())


class TFHubUpdateOpsTest(tf.test.TestCase):

  def testUpdateOps(self):
    spec = hub.create_module_spec(update_ops_module_fn)
    with tf_v1.Session() as sess:
      trainable_module = hub.Module(spec, trainable=True)
      fixed_module = hub.Module(spec, trainable=False)

      # TODO(b/62433105): Understand what is the desired behaviour of UPDATE_OPS
      # and applying a Module multiple times. For now UPDATE_OPS probably only
      # do something reasonable if each Module is applied exactly one time.
      trainable_module()
      fixed_module()

      variable = tf.Variable(0.0)
      step = tf.Variable(0, trainable=False, name="global_step")
      update_ops = tf_v1.get_collection(tf_v1.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = tf_v1.train.GradientDescentOptimizer(0.1).minimize(
            loss=variable,
            global_step=step)

      sess.run(tf_v1.global_variables_initializer())
      sess.run(train_op)
      trainable_module_vars = list(trainable_module.variable_map.values())
      self.assertEqual(len(trainable_module_vars), 1)
      self.assertEqual(sess.run(trainable_module_vars[0]), 1)
      fixed_module_vars = list(fixed_module.variable_map.values())
      self.assertEqual(len(fixed_module_vars), 1)
      self.assertEqual(sess.run(fixed_module_vars[0]), 0)


def batch_norm_module_fn(is_training):
  """Module that exercises batch normalization, incl. UPDATE_OPS."""
  x = tf_v1.placeholder(dtype=tf.float32, shape=[None, 1], name="x")
  y = tf_v1.layers.batch_normalization(
      momentum=0.4,
      inputs=x,
      fused=False,
      training=is_training)
  hub.add_signature(inputs=x, outputs=y)


class TFHubBatchNormModuleTest(tf.test.TestCase):

  # This test is intended to verify the following:
  # 1) A module_fn that uses batch normalization through tf.layers.contrib
  #    (and its underlying utilities from tf.nn) can be used to create,
  #    export, load and use the Module.
  # 2) Batch normalization learns the scale and offset parameters for its
  #    output as it should.
  # 3) The UPDATE_OPS added internally for the moving_mean and moving_variance
  #    over the training data are properly executed at training time, and their
  #    results are used at serving time, without further change.
  def testModuleWithBatchNorm(self):
    export_path = os.path.join(self.get_temp_dir(), "batch-norm-module")
    # This test resorts to lookup by name to retrieve the moving mean,
    # because tf.contrib.layers.batch_norm() does not return it, and even if,
    # module_fn() has no way to return it next to the result for training.
    moving_mean_name = (
        "module/batch_normalization/moving_mean/Read/ReadVariableOp:0")

    batch_norm_train_tags = ["batch_norm_trains"]
    batch_norm_fixed_tags = ["batch_norm_fixed"]
    spec = hub.create_module_spec(
        batch_norm_module_fn,
        [(batch_norm_train_tags, {"is_training": True}),
         (batch_norm_fixed_tags, {"is_training": False})])
    # Test Module creation and training.
    with tf.Graph().as_default() as g:
      m = hub.Module(spec, trainable=True, tags=batch_norm_train_tags)
      # The module is trained on a fixed batch of inputs, which has a mean
      # of 12.0 and some sample variance of a less obvious value. The module
      # learns scale and offset parameters that achieve the mapping x --> 2*x
      # for the observed mean and variance.
      x = tf.constant([[11.0], [12.0], [13.0]])
      training_mean = [12.0]
      y_target = tf.constant([[22.0], [24.0], [26.0]])
      y = m(x)
      step = tf.Variable(0, trainable=False, name="global_step")
      moving_mean = g.get_tensor_by_name(moving_mean_name)
      update_ops = tf_v1.get_collection(tf_v1.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train = tf_v1.train.GradientDescentOptimizer(0.1).minimize(
            loss=tf_v1.losses.mean_squared_error(y, y_target), global_step=step)
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllClose(sess.run(moving_mean), [0.0])
        for _ in range(100):
          sess.run([train])
        trained_moving_mean, trained_y = sess.run([moving_mean, y])
        self.assertAllClose(trained_moving_mean, training_mean)
        self.assertAllClose(trained_y, [[22.0], [24.0], [26.0]])
        # Test export.
        m.export(export_path, sess)

    # Test import and use.
    spec = load_module_spec(export_path)
    with tf.Graph().as_default() as g:
      # The module gets run for inference on inputs with different mean and
      # variance. However, both mean and variance as well as offset and scale
      # are now frozen to the values from learning, so the same mapping
      # x --> 2*x is recovered.
      x = tf.constant([[10.0], [20.0], [30.0]])
      y = hub.Module(
          spec, tags=batch_norm_fixed_tags)(x)
      moving_mean = g.get_tensor_by_name(moving_mean_name)
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        for _ in range(100):
          served_moving_mean, served_y = sess.run([moving_mean, y])
        # No update occurs to the moving_mean from training time.
        self.assertAllClose(served_moving_mean, training_mean)
        # Prediction results are correct.
        self.assertAllClose(served_y, [[20.0], [40.0], [60.0]])


def multiple_outputs_module_fn():
  x = tf_v1.placeholder(dtype=tf.float32)
  v = tf.Variable([3.0])
  hub.add_signature(
      inputs={"x": x},
      outputs={"y": v * x, "z": v * v * x})


class TFHubMultipleOutputsTest(tf.test.TestCase):

  def testMultipleOutputs(self):
    with tf_v1.Session() as sess:
      spec = hub.create_module_spec(multiple_outputs_module_fn)
      m = hub.Module(spec)
      output = m(tf.constant([2.0]), as_dict=True)
      output1 = output["y"]
      output2 = output["z"]
      sess.run(tf_v1.global_variables_initializer())
      self.assertAllClose(sess.run(output1), [6.0])
      self.assertAllClose(sess.run(output2), [18.0])


def create_assets_module_fn(vocabulary_file):

  def assets_module_fn():
    indices = tf_v1.placeholder(dtype=tf.int64, name="indices")
    outputs = do_table_lookup(indices, vocabulary_file)
    hub.add_signature(inputs=indices, outputs=outputs)

  return assets_module_fn


def create_consumer_module_fn(exported_hub_module):

  def consumer_module_fn():
    indices = tf_v1.placeholder(dtype=tf.int64, name="indices")
    inner_module = hub.Module(exported_hub_module)
    inner_module_output = inner_module(indices)
    output = tf.identity(inner_module_output)
    hub.add_signature(inputs=indices, outputs=output)

  return consumer_module_fn


class TFHubAssetsTest(tf.test.TestCase):

  def create_vocab_file(self, path, vocab):
    vocabulary_file = os.path.join(self.get_temp_dir(), "tokens.txt")
    with open(vocabulary_file, "w+") as vocab_file:
      for line in vocab:
        vocab_file.write(line)
        vocab_file.write(os.linesep)
    return vocabulary_file

  def testAssets(self):
    export_path = os.path.join(self.get_temp_dir(), "assets-module")
    vocabulary_file = self.create_vocab_file("tokens.txt",
                                             ["emerson", "lake", "palmer"])
    with tf.Graph().as_default():
      assets_module_fn = create_assets_module_fn(vocabulary_file)
      spec = hub.create_module_spec(assets_module_fn)
      embedding_module = hub.Module(spec)
      output = embedding_module(tf.constant([1, 2], dtype=tf.int64))
      with tf_v1.Session() as sess:
        sess.run(tf_v1.tables_initializer())
        self.assertAllEqual(list(sess.run(output)), [b"lake", b"palmer"])
        embedding_module.export(export_path, sess)

    asset_file = os.path.join(*[export_path, "assets", "tokens.txt"])
    # Check that asset file got written to the expected place:
    self.assertTrue(tf_v1.gfile.Exists(asset_file))

    # Assets should be hermetic, so we can delete the original vocab file:
    tf_v1.gfile.Remove(vocabulary_file)

    with tf.Graph().as_default():
      spec = load_module_spec(export_path)
      embedding_module = hub.Module(spec)
      output = embedding_module(tf.constant([1, 2], dtype=tf.int64))
      with tf_v1.Session() as sess:
        sess.run(tf_v1.tables_initializer())
        # Check functionality:
        self.assertAllEqual(list(sess.run(output)), [b"lake", b"palmer"])
        # Check that the ASSET_FILEPATHS collection was restored properly:
        asset_filepaths = [
            sess.run(tensor)
            for tensor in tf_v1.get_collection(tf_v1.GraphKeys.ASSET_FILEPATHS)
        ]
        # ASSET_FILEPATHS are added for the state graph and for the apply graph:
        self.assertAllEqual(asset_filepaths,
                            [tf.compat.as_bytes(asset_file)] * 2)

  def testDuplicateAssetCopy(self):
    export_path = os.path.join(self.get_temp_dir(), "assets-module")

    def module_with_duplicate_asset():
      vocabulary_file = self.create_vocab_file("tokens2.txt", ["1", "2", "3"])
      indices1 = tf_v1.placeholder(dtype=tf.int64, name="indices1")
      indices2 = tf_v1.placeholder(dtype=tf.int64, name="indices2")
      hub.add_signature(
          inputs={
              "indices_1": indices1,
              "indices_2": indices2,
          },
          outputs={
              "x": do_table_lookup(indices1, vocabulary_file),
              "y": do_table_lookup(indices2, vocabulary_file),
          })

    with tf.Graph().as_default():
      spec = hub.create_module_spec(module_with_duplicate_asset)
      module_a = hub.Module(spec)
      module_a({"indices_1": tf.constant([1, 2], dtype=tf.int64),
                "indices_2": tf.constant([1, 2], dtype=tf.int64)}, as_dict=True)
      with tf_v1.Session() as sess:
        sess.run(tf_v1.tables_initializer())
        module_a.export(export_path, sess)

  def testExportedConsumerModelWorksIfItUsesHubModuleWithAssets(self):
    # 1. Create and export a module with assets.
    module_export_path = os.path.join(self.get_temp_dir(), "small-module")
    vocabulary_file = self.create_vocab_file("tokens.txt",
                                             ["emerson", "lake", "palmer"])
    assets_module_fn = create_assets_module_fn(vocabulary_file)
    spec = hub.create_module_spec(assets_module_fn)
    with tf.Graph().as_default():
      small_module = hub.Module(spec)
      with tf_v1.Session() as sess:
        small_module.export(module_export_path, sess)
    # 2. Remove the original vocab file and move the module to another location.
    tf_v1.gfile.Remove(vocabulary_file)
    inner_module_path = os.path.join(self.get_temp_dir(), "inner-module")
    tf_v1.gfile.Rename(module_export_path, inner_module_path)
    del module_export_path
    # 3. Use the module in a consumer model (which is another module here).
    module_export_path = os.path.join(self.get_temp_dir(), "consumer-module")
    consumer_module_fn = create_consumer_module_fn(inner_module_path)
    spec = hub.create_module_spec(consumer_module_fn)
    with tf.Graph().as_default():
      consumer_module = hub.Module(spec)
      with tf_v1.Session() as sess:
        consumer_module.export(module_export_path, sess)
    # 4. Delete the inner module on disk and move the consumer model to a final
    # location for serving.
    tf_v1.gfile.DeleteRecursively(inner_module_path)
    module_serving_path = os.path.join(self.get_temp_dir(), "serving-module")
    tf_v1.gfile.Rename(module_export_path, module_serving_path)
    # 5. Make sure the model can be served successfully.
    with tf.Graph().as_default():
      serving_module = hub.Module(module_serving_path)
      output = serving_module(tf.constant([1, 2], dtype=tf.int64))
      with tf_v1.Session() as sess:
        sess.run(tf_v1.tables_initializer())
        self.assertAllEqual(list(sess.run(output)), [b"lake", b"palmer"])


def another_stateful_module_fn():
  """Stateful module with inputs."""
  module_input = tf_v1.placeholder(dtype=tf.float32)
  variable = tf.Variable([3.0], name="iamtheoneandonly")
  hub.add_signature(inputs=module_input, outputs=module_input*variable)


class TFHubApplyStatefulModuleMultipleTimesTest(tf.test.TestCase):

  def testApplyStatefulModuleMultipleTimes(self):
    export_path = os.path.join(self.get_temp_dir(), "another-module")

    with tf_v1.Session() as sess:
      spec = hub.create_module_spec(another_stateful_module_fn)
      stateful_module = hub.Module(spec, trainable=True)
      times2 = stateful_module(tf.constant([2.0]))
      times3 = stateful_module(tf.constant([3.0]))
      step = tf.Variable(0, trainable=False, name="global_step")
      # Training will adapt the hidden variable to be approximately 2:
      train = tf_v1.train.GradientDescentOptimizer(0.05).minimize(
          loss=tf_v1.losses.mean_squared_error(times2, [4.0]),
          global_step=step)

      sess.run(tf_v1.global_variables_initializer())
      for _ in range(50):
        sess.run(train)
      self.assertAllClose(sess.run(times2), [4.0])
      self.assertAllClose(sess.run(times3), [6.0])
      stateful_module.export(export_path, sess)
    with tf_v1.Session() as sess:
      stateful_module = hub.Module(export_path)
      times4 = stateful_module(tf.constant([4.0]))
      times5 = stateful_module(tf.constant([5.0]))
      sess.run(tf_v1.global_variables_initializer())
      self.assertAllClose(sess.run(times4), [8.0])
      self.assertAllClose(sess.run(times5), [10.0])

  def testMultipleApplicationsInDifferentScopes(self):
    with tf.Graph().as_default():
      export_path = os.path.join(self.get_temp_dir(), "module-applied-in-scope")

      spec = hub.create_module_spec(another_stateful_module_fn)
      stateful_module = hub.Module(spec, name="moduleA")
      with tf.name_scope("foo"):
        with tf_v1.variable_scope("bar"):
          times2 = stateful_module(tf.constant([2.0]))
      with tf.name_scope("baz"):
        times3 = stateful_module(tf.constant([3.0]))

      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllClose(sess.run(times2), [6.0])
        self.assertAllClose(sess.run(times3), [9.0])
        self.assertEqual(len(stateful_module.variable_map), 1)
        self.assertEqual(
            stateful_module.variable_map["iamtheoneandonly"].name,
            "moduleA/iamtheoneandonly:0")
        stateful_module.export(export_path, sess)

      # Check minimal functionality of the exported module.
    with tf.Graph().as_default():
      stateful_module = hub.Module(export_path, name="moduleB")
      times2 = stateful_module(tf.constant([2.0]))
      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        self.assertAllClose(sess.run(times2), [6.0])


def multiple_signature_module_fn():
  """Stateful module with multiple signatures."""
  weight = tf.Variable([3.0])

  x_input = tf_v1.placeholder(dtype=tf.float32)
  x_output = tf.multiply(x_input, weight)
  hub.add_signature("mul", inputs=x_input, outputs=x_output)

  y_input = tf_v1.placeholder(dtype=tf.float32)
  y_output = tf.divide(y_input, weight)
  hub.add_signature("div", inputs=y_input, outputs=y_output)


class TFHubModuleWithMultipleSignatures(tf.test.TestCase):

  def testGetSignatures(self):
    spec = hub.create_module_spec(multiple_signature_module_fn)
    self.assertEqual(sorted(spec.get_signature_names()), ["div", "mul"])

  def testModuleWithMultipleSignatures(self):
    with tf.Graph().as_default():
      spec = hub.create_module_spec(multiple_signature_module_fn)
      module_a = hub.Module(spec, name="moduleA")
      in_tensor = tf_v1.placeholder(dtype=tf.float32)
      out_tensor_a = module_a(in_tensor, signature="mul")
      out_tensor_b = module_a(out_tensor_a, signature="div")

      with tf_v1.Session() as sess:
        sess.run(tf_v1.global_variables_initializer())
        in_values = [6, 3, 1]
        self.assertAllClose(
            sess.run(out_tensor_b, feed_dict={in_tensor: in_values}), in_values)


def cond_module_fn():
  """Computes relu(x) with a conditional."""
  x = tf_v1.placeholder(dtype=tf.float32, name="x", shape=[])
  result = tf.cond(0 < x, lambda: tf.identity(x), lambda: tf.constant(0.0))
  hub.add_signature(inputs=x, outputs=result)


def nested_cond_module_fn():
  """Computes relu(x) with nested conditionals."""
  x = tf_v1.placeholder(dtype=tf.float32, name="x", shape=[])
  # pylint: disable=g-long-lambda
  result = tf.cond(
      0 < x,
      lambda: tf.cond(3 < x,
                      lambda: tf.identity(x),
                      lambda: tf.multiply(x, 1.0)),
      lambda: tf.cond(x < -3,
                      lambda: tf.constant(0.0),
                      lambda: tf.multiply(0.0, 1.0)))
  # pylint: enable=g-long-lambda
  hub.add_signature(inputs=x, outputs=result)


def while_module_fn():
  """Compute x^n with while_loop."""
  x = tf_v1.placeholder(dtype=tf.float32, name="x", shape=[])
  n = tf_v1.placeholder(dtype=tf.int32, name="n")
  _, pow_x = tf.while_loop(
      lambda i, ix: i < n, lambda i, ix: [tf.add(i, 1), ix * x],
      [tf.constant(0), tf.constant(1.0)])
  hub.add_signature(inputs={"x": x, "n": n}, outputs=pow_x)


def nested_control_flow_module_fn():
  """Compute the sum of elements greater than 'a' with nested control flow."""
  elems = tf_v1.placeholder(
      dtype=tf.float32, name="elems", shape=[None])
  a = tf_v1.placeholder(dtype=tf.float32, name="a")

  def sum_above_a(acc, x):
    return acc + tf.cond(x > a, lambda: x, lambda: 0.0)

  hub.add_signature(
      inputs={"elems": elems, "a": a},
      outputs=tf.foldl(sum_above_a, elems, initializer=tf.constant(0.0)))


class TFHubModulesWithControlFlow(tf.test.TestCase):

  def _testCondModule(self):
    self._testReluModule(cond_module_fn)

  def testCondModule(self):
    self._testCondModule()

  @test_util.enable_control_flow_v2
  def testCondModuleWithControlFlowV2(self):
    self._testCondModule()

  def _testModuleWithNestedConds(self):
    self._testReluModule(nested_cond_module_fn)

  def testModuleWithNestedConds(self):
    self._testModuleWithNestedConds()

  @test_util.enable_control_flow_v2
  def testModuleWithNestedCondsWithControlFlowV2(self):
    self._testModuleWithNestedConds()

  def _testReluModule(self, module_fn):
    spec = hub.create_module_spec(module_fn)
    with tf.Graph().as_default():
      with tf_v1.Session() as sess:
        x = tf_v1.placeholder(dtype=tf.float32, name="x")
        relu_module = hub.Module(spec)
        y = relu_module(x)
        grad = tf.gradients([y], [x])
        self.assertAllClose(sess.run(y, {x: 9.1}), 9.1)
        self.assertAllClose(sess.run(y, {x: -2.4}), 0.0)
        self.assertAllClose(sess.run(grad, {x: 2}), [1.0])
        self.assertAllClose(sess.run(grad, {x: -2}), [0.0])

  def _testWhileModule(self):
    spec = hub.create_module_spec(while_module_fn)
    with tf.Graph().as_default():
      with tf_v1.Session() as sess:
        x = tf_v1.placeholder(tf.float32)
        n = tf_v1.placeholder(tf.int32)
        pow_module = hub.Module(spec)
        y = pow_module({"x": x, "n": n})
        grad = tf.gradients([y], [x])
        self.assertAllClose(sess.run(y, {x: 9.1, n: 1}), 9.1)
        self.assertAllClose(sess.run(y, {x: 2.4, n: 2}), 5.76)
        self.assertAllClose(sess.run(grad, {x: 2, n: 3}), [12.0])

  def testWhileModule(self):
    self._testWhileModule()

  @test_util.enable_control_flow_v2
  def testWhileModuleWithControlFlowV2(self):
    self._testWhileModule()

  @test_util.run_v1_only("b/138681007")
  def testUseModuleWithWhileLoopInsideCond(self):
    spec = hub.create_module_spec(while_module_fn)
    with tf.Graph().as_default():
      m = hub.Module(spec)
      cond = tf.cond(
          tf.equal(tf.constant(0), tf.constant(0)),
          lambda: m({"x": tf.constant(3.0), "n": tf.constant(2)}),
          lambda: tf.constant(4.0))
      with tf_v1.Session() as sess:
        self.assertEqual(sess.run(cond), 9.0)

  def _testNestedControlFlowModule(self):
    spec = hub.create_module_spec(nested_control_flow_module_fn)
    with tf.Graph().as_default():
      with tf_v1.Session() as sess:
        elems = tf_v1.placeholder(tf.float32, shape=[None])
        a = tf_v1.placeholder(tf.float32)
        m = hub.Module(spec)
        out = m({"elems": elems, "a": a})
        grad = tf.gradients([out], [elems])
        self.assertAllClose(
            sess.run(out, {
                a: 1.1,
                elems: [10, 0, 0.5, 1.2]
            }), 11.2)

        self.assertAllClose(sess.run(grad, {a: 1, elems: [10, 0, 0.5, 1.2]}),
                            [[1.0, 0.0, 0.0, 1.0]])

  def testNestedControlFlowModule(self):
    self._testNestedControlFlowModule()

  @test_util.enable_control_flow_v2
  def testNestedControlFlowModuleWithControlFlowV2(self):
    self._testNestedControlFlowModule()


def attached_messages_module_fn(tagged=0):
  x = tf_v1.placeholder(tf.float32, shape=[None])
  hub.add_signature(inputs={"x": x}, outputs={"y": 2*x})
  # For brevity, this test borrows two well-known, stable message types
  # from TensorFlow. They are not likely choices for actual uses.
  hub.attach_message("numbers",
                     tf_v1.train.Int64List(value=[-3]))  # Overwritten.
  hub.attach_message("numbers", tf_v1.train.Int64List(value=[42, 69]))
  hub.attach_message("letters", tf_v1.train.BytesList(value=[
      tf.compat.as_bytes("abc"), tf.compat.as_bytes("xyz")]))
  hub.attach_message("tagged", tf_v1.train.Int64List(value=[tagged]))


class TFHubModuleWithAttachedMessages(tf.test.TestCase):

  def testModuleSpec(self):
    """This is the general test for ModuleSpec and native_module._ModuleSpec."""
    spec = hub.create_module_spec(attached_messages_module_fn)
    attached_letters = spec.get_attached_message("letters",
                                                 tf_v1.train.BytesList)
    self.assertSequenceEqual(
        attached_letters.value,
        [tf.compat.as_bytes("abc"),
         tf.compat.as_bytes("xyz")])
    attached_numbers = spec.get_attached_message("numbers",
                                                 tf_v1.train.Int64List)
    self.assertSequenceEqual(attached_numbers.value, [42, 69])
    attached_train = spec.get_attached_message("tagged", tf_v1.train.Int64List)
    self.assertSequenceEqual(attached_train.value, [0])
    self.assertIsNone(spec.get_attached_message("bad", tf_v1.train.BytesList))
    with self.assertRaises(KeyError):
      spec.get_attached_message("bad", tf_v1.train.BytesList, required=True)

  def testModule(self):
    """Tests forwarding from Module to ModuleSpec."""
    spec = hub.create_module_spec(attached_messages_module_fn)
    with tf.Graph().as_default():
      module = hub.Module(spec)
      attached = module.get_attached_message("numbers", tf_v1.train.Int64List)
      self.assertSequenceEqual(attached.value, [42, 69])

  def testGraphVersions(self):
    """Tests native_module._ModuleSpec for explicit tags arguments."""
    tags_and_args = [(set(), {"tagged": 1}),
                     ({"double", "the", "value"}, {"tagged": 2})]
    spec = hub.create_module_spec(attached_messages_module_fn,
                                  tags_and_args=tags_and_args)
    for tags, args in tags_and_args:
      attached_to_spec = spec.get_attached_message(
          "tagged", tf_v1.train.Int64List, tags=tags)
      self.assertSequenceEqual(attached_to_spec.value, [args["tagged"]])
      with tf.Graph().as_default():
        module = hub.Module(spec, tags=tags)
        attached_to_module = module.get_attached_message(
            "tagged", tf_v1.train.Int64List)
        self.assertSequenceEqual(attached_to_module.value, [args["tagged"]])

  def testSeparateCopies(self):
    """Mutating returned objects does not affect future returned values."""
    spec = hub.create_module_spec(attached_messages_module_fn)
    attached_numbers = spec.get_attached_message("numbers",
                                                 tf_v1.train.Int64List)
    self.assertSequenceEqual(attached_numbers.value, [42, 69])
    attached_numbers.Clear()
    self.assertSequenceEqual(attached_numbers.value, [])
    attached_numbers = spec.get_attached_message("numbers",
                                                 tf_v1.train.Int64List)
    self.assertSequenceEqual(attached_numbers.value, [42, 69])


class TFHubOpsTest(tf.test.TestCase):

  def testRegisterLinkedOpsError(self):
    with self.assertRaisesRegexp(tf.errors.NotFoundError, "non-existent-op"):
      native_module.register_ops_if_needed({"non-existent-op"})


class TFHubExportSpecTest(tf.test.TestCase):

  def f(self, x, dim=10):
    return tf_v1.layers.dense(x, dim)

  def module_fn(self, dim=10):
    x = tf_v1.placeholder(dtype=tf.float32, shape=[None, dim])
    y = self.f(x, dim=dim)
    hub.add_signature(inputs=x, outputs=y)

  def createCheckpoint(self, scope=None):
    checkpoint_path = os.path.join(self.get_temp_dir(), "model")
    with tf.Graph().as_default():
      x = tf_v1.get_variable(
          "x", [32, 10], initializer=tf_v1.initializers.random_normal())
      if scope:
        with tf_v1.variable_scope(scope):
          y = self.f(x)
      else:
        y = self.f(x)
      tf_v1.layers.dense(y, 20)

      saver = tf_v1.train.Saver()
      init_op = tf_v1.initializers.global_variables()

      with tf_v1.Session() as session:
        session.run(init_op)
        saver.save(session, checkpoint_path)

    return checkpoint_path

  def testExportModuleSpec(self):
    checkpoint_path = self.createCheckpoint()
    export_path = os.path.join(self.get_temp_dir(), "module1")

    spec = hub.create_module_spec(self.module_fn)
    spec.export(export_path,
                checkpoint_path=checkpoint_path)

  def testExportModuleSpec_withWrongShape(self):
    checkpoint_path = self.createCheckpoint(scope="block")
    export_path = os.path.join(self.get_temp_dir(), "module2")

    spec = hub.create_module_spec(lambda: self.module_fn(dim=20))
    with self.assertRaisesRegexp(ValueError, "doesn't match with shape of"):
      spec.export(export_path,
                  checkpoint_path=checkpoint_path,
                  name_transform_fn=lambda x: "block/" + x)

  def testExportModuleSpec_withWrongScope(self):
    checkpoint_path = self.createCheckpoint("block2")
    export_path = os.path.join(self.get_temp_dir(), "module3")

    spec = hub.create_module_spec(self.module_fn)
    with self.assertRaisesRegexp(ValueError, "bias is not found in"):
      spec.export(export_path,
                  checkpoint_path=checkpoint_path,
                  name_transform_fn=lambda x: "block/" + x)


class TFHubUsageWithEager(tf.test.TestCase):

  def testWrapFunction(self):
    if not tf.executing_eagerly():
      self.skipTest("Test requires eager.")

    spec = hub.create_module_spec(stateful_rv_with_input_module_fn)

    initializers = []
    def use_module(x, y):
      m = hub.Module(spec, name="module_", trainable=True)
      initializers.append(tf_v1.initializers.global_variables())
      return [m(x), m(y)]

    input_signature = [
        tf.TensorSpec((), tf.float32),
        tf.TensorSpec((), tf.float32),
    ]

    f = tf_v1.wrap_function(use_module, input_signature)
    f.prune([], initializers)()
    self.assertAllEqual(
        [x.numpy() for x in f(9.0, 6.0)],
        [19.0, 16.0])


if __name__ == "__main__":
  tf.test.main()
