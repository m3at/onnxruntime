// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/helper.h"

#include <algorithm>
#include <vector>

#ifdef __APPLE__
#include <sys/utsname.h>
#include <TargetConditionals.h>
#endif

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/model/host_utils.h"

namespace onnxruntime {
namespace coreml {

namespace {
bool GetShapeImpl(const NodeArg& node_arg, std::vector<int64_t>& shape_out, const logging::Logger& logger,
                  bool allow_dynamic_shape) {
  const auto* shape_proto = node_arg.Shape();
  if (!shape_proto) {
    LOGS(logger, WARNING) << "NodeArg [" << node_arg.Name() << "] has no shape info";
    return false;
  }

  std::vector<int64_t> shape{};
  shape.reserve(shape_proto->dim().size());

  for (int i = 0; i < shape_proto->dim().size(); ++i) {
    const auto& dim = shape_proto->dim(i);
    if (utils::HasDimValue(dim)) {
      shape.push_back(dim.dim_value());
    } else {
      // dynamic dimension
      if (!allow_dynamic_shape) {
        LOGS(logger, WARNING) << "NodeArg [" << node_arg.Name() << "] has shape with dynamic dimension";
        return false;
      }
      shape.push_back(-1);
    }
  }

  shape_out = std::move(shape);
  return true;
}
}  // namespace

bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger) {
  return GetShapeImpl(node_arg, shape, logger, /* allow_dynamic_shape */ true);
}

bool GetStaticShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger) {
  return GetShapeImpl(node_arg, shape, logger, /* allow_dynamic_shape */ false);
}

bool IsStaticShape(gsl::span<const int64_t> shape) {
  return std::find(shape.begin(), shape.end(), int64_t{-1}) == shape.end();
}

bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer, const logging::Logger& logger) {
  const auto& op_builders = GetOpBuilders();
  if (Contains(op_builders, node.OpType())) {
    const auto* op_builder = op_builders.at(node.OpType());
    OpBuilderInputParams input_params(graph_viewer);
    return op_builder->IsOpSupported(node, input_params, logger);
  } else {
    return false;
  }
}

bool IsInputSupported(const NodeArg& input, const std::string& parent_name, const logging::Logger& logger) {
  if (!input.Exists()) {
    // optional input that is not provided
    return true;
  }

  const auto& input_name = input.Name();
  const auto* shape_proto = input.Shape();
  // We do not support input with no shape
  if (!shape_proto) {
    LOGS(logger, VERBOSE) << "Input [" << input_name << "] of [" << parent_name
                          << "] has no shape";
    return false;
  }

  for (const auto& dim : shape_proto->dim()) {
    // For some undocumented reason, Apple CoreML framework will fail loading the model if the model
    // input has dimension > 16384
    // See this issue, https://github.com/apple/coremltools/issues/1003
    if (utils::HasDimValue(dim) && dim.dim_value() > 16384) {
      LOGS(logger, WARNING) << "CoreML does not support input dim > 16384, input:" << input_name
                            << ", actual dim: " << dim.dim_value();
      return false;
    }
  }

  return true;
}

std::unordered_set<const Node*> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                  const logging::Logger& logger) {
  std::unordered_set<const Node*> supported_nodes{};

#ifdef __APPLE__
  if (!util::HasRequiredBaseOS()) {
    LOGS(logger, WARNING) << "All ops will fallback to CPU EP, because we do not have supported OS";
    return supported_nodes;
  }
#endif

  for (const auto& node : graph_viewer.Nodes()) {
    const bool supported = IsNodeSupported(node, graph_viewer, logger);
    LOGS(logger, VERBOSE) << "Operator type: [" << node.OpType()
                          << "] index: [" << node.Index()
                          << "] name: [" << node.Name()
                          << "] supported: [" << supported
                          << "]";
    if (supported) {
      supported_nodes.insert(&node);
    }
  }

  return supported_nodes;
}

bool HasNeuralEngine(const logging::Logger& logger) {
  bool has_neural_engine = false;

#ifdef __APPLE__
  struct utsname system_info;
  uname(&system_info);
  LOGS(logger, VERBOSE) << "Current Apple hardware info: " << system_info.machine;

#if TARGET_OS_IPHONE
  // utsname.machine has device identifier. For example, identifier for iPhone Xs is "iPhone11,2".
  // Since Neural Engine is only available for use on A12 and later, major device version in the
  // identifier is checked for these models:
  // A12: iPhone XS (11,2), iPad Mini - 5th Gen (11,1)
  // A12X: iPad Pro - 3rd Gen (8,1)
  // For more information, see https://www.theiphonewiki.com/wiki/Models
  size_t str_len = strnlen(system_info.machine, onnxruntime::kMaxStrLen);
  if (str_len > 4 && strncmp("iPad", system_info.machine, 4) == 0) {
    const int major_version = atoi(system_info.machine + 4);
    has_neural_engine = major_version >= 8;  // There are no device between iPad 8 and 11.
  } else if (str_len > 6 && strncmp("iPhone", system_info.machine, 6) == 0) {
    const int major_version = atoi(system_info.machine + 6);
    has_neural_engine = major_version >= 11;
  }
#elif TARGET_OS_OSX && TARGET_CPU_ARM64
  // Only Mac with arm64 CPU (Apple Silicon) has ANE.
  has_neural_engine = true;
#endif  // #if TARGET_OS_IPHONE
#else
  // In this case, we are running the EP on non-apple platform, which means we are running the model
  // conversion with CoreML EP enabled, for this we always assume the target system has Neural Engine
  LOGS(logger, VERBOSE) << "HasNeuralEngine running on non-Apple hardware for model conversion only";
  has_neural_engine = true;
#endif  // #ifdef __APPLE__

  return has_neural_engine;
}

}  // namespace coreml
}  // namespace onnxruntime
