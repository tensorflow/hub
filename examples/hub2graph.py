#!/usr/bin/env python3
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""
Fetch a module from Hub and write it as GraphDef binary Protocol Buffers file
and also optionally as Tensorboard logs. The former is useful for
running inference in other languages with lagging API support, e.g. Go.
The latter allows to visualize the model in Tensorboard.

Usage:

```
python3 /path/to/hub2graph.py -m <TF Hub path or URL> -o output_graph_def.pb
```

This script is not intended to digest large (over 1GB) graphs.
The hard limit is 2 GB - the maximum size of Protocol Buffers.
"""

import argparse
import logging
import sys
import os

import tensorflow as tf
from tensorflow.python.framework import graph_util, graph_io
from tensorflow.python.summary import summary
import tensorflow_hub as tfhub


def setup():
    """
    Parse command line arguments and setup logging.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--hub", required=True,
                        help="Tensorflow Hub module URL.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output graph definition.")
    parser.add_argument("-t", "--tensorboard",
                        help="Tensorboard output log directory.")
    parser.add_argument("--disable-optimization", action="store_true",
                        help="Do not exclude nodes unrelated to inference.")
    parser.add_argument("-v", "--log-level", choices=logging._nameToLevel,
                        default="WARN")
    args = parser.parse_args()
    level = logging._nameToLevel[args.log_level]
    logging.basicConfig(level=level)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(level // 10 - 2)
    return args


def tensor_name(t):
    """
    Prepends "module" to tensor name and removes ":0" in the end
    """
    return "module/" + t.name.rsplit(":", 1)[0]


def print_signature(sig, name):
    """
    Prints the Markdown table with inputs or outputs of a Hub module.
    """
    label_size = max([len(k) for k in sig])
    name_size = max([len(tensor_name(v)) for _, v in sig.items()])
    print("\n%s\n%s\n\n| label%s | tensor%s |\n|:------%s|:-------%s|" % (
        name,
        "=" * len(name),
        " " * (label_size - 5) if label_size > 5 else "",
        " " * (name_size - 6),
        "-" * (label_size - 5) if label_size > 5 else "",
        "-" * (name_size - 6)
    ))
    for k, v in sorted(sig.items()):
        print("| %s%s | %s%s |" % (
            k, " " * (label_size - len(k)),
            tensor_name(v), " " * (name_size - len(tensor_name(v)))
        ))
    print()


def main():
    """
    Performs the conversion from Hub module to GraphDef + TB.
    """
    args = setup()
    log = logging.getLogger("main")
    with tf.Graph().as_default() as graph:
        # everything happens inside the single default graph
        log.info("Importing %s", args.hub)
        module = tfhub.Module(args.hub)
        # protected members are used here
        # `outputs = module(inputs)` has several problems:
        # 1. slow
        # 2. creates a new subgraph which is hard to rename and unwrap
        # 3. creates redundant placeholders for our goal
        signature_def = module._impl._meta_graph.signature_def.get(
            module._impl.get_signature_name(None))
        print_signature(signature_def.inputs, "Inputs")
        print_signature(signature_def.outputs, "Outputs")
        input_names = [
            tensor_name(v) for _, v in sorted(signature_def.inputs.items())]
        output_names = [
            tensor_name(v) for _, v in sorted(signature_def.outputs.items())]
        log.info("Launching the session")
        with tf.Session(graph=graph) as session:
            log.info("Initializing variables")
            session.run(tf.global_variables_initializer())
            graph_def = graph.as_graph_def()
            if not args.disable_optimization:
                nodes = len(graph_def.node)
                graph_def = graph_util.extract_sub_graph(
                    graph_def, input_names + output_names)
                log.info("Optimized the graph: %d -> %d nodes", nodes,
                         len(graph_def.node))
            log.info("Preparing the exported graph")
            # reset the devices
            for node in graph_def.node:
                node.device = ""
            # turn variables into constants
            constant_graph = graph_util.convert_variables_to_constants(
                session, graph_def, output_names)
            log.info("Writing %s", args.output)
            graph_io.write_graph(constant_graph, *os.path.split(args.output),
                                 as_text=False)
            if args.tensorboard:
                # Using the private API here because
                # add_graph(GraphDef) is deprecated;
                # tf.import_graph_def() places the graph inside "import"
                # and there is no way to avoid that, we would also have to
                # destroy the current graph and initialize a new one.
                summary.FileWriter(args.tensorboard)._add_graph_def(
                    constant_graph)
                print("\nVisualize: tensorboard --logdir=" + args.tensorboard)


if __name__ == "__main__":
    sys.exit(main())
