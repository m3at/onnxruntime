// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>

#ifdef IO_BUFFER_ENABLED
#include <gpu/gpu_context_api_ocl.hpp>
#include <gpu/gpu_config.hpp>
#endif

#include "core/providers/shared_library/provider_api.h"
#include "../backend_utils.h"
#include <ngraph/frontend/onnx_import/onnx.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include "basic_backend.h"
#include "../backend_manager.h"

namespace onnxruntime {

namespace openvino_ep {

using namespace backend_utils;

BasicBackend::BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
                           GlobalContext& global_context,
                           const SubGraphContext& subgraph_context)
    : global_context_(global_context), subgraph_context_(subgraph_context) {
  std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
  bool vpu_status = false;
  std::string model_blob_name;
  std::string ov_compiled_blobs_dir = "";

#if (defined OPENVINO_2021_2) || (defined OPENVINO_2021_3)
  //Flow for OpenVINO versions supported till OV 2021.3
  bool import_blob_status = false;
  PopulateCompiledDirectory(hw_target, ov_compiled_blobs_dir, model_blob_name, vpu_status);

  if(!openvino_ep::BackendManager::GetGlobalContext().is_wholly_supported_graph) {
    ie_cnn_network_ = CreateCNNNetwork(model_proto, global_context_, subgraph_context_, const_outputs_map_);
    SetIODefs(model_proto, ie_cnn_network_, subgraph_context_.output_names, const_outputs_map_, global_context_.device_type);
    //Validate Constant subgraphs
    if (ValidateSubgraph(const_outputs_map_))
      return;
  }

  if (vpu_status == true || openvino_ep::backend_utils::UseCompiledNetwork()) {
    const std::string model_blob_path = ov_compiled_blobs_dir + "/" + model_blob_name;
    const std::string compiled_blob_path = onnxruntime::GetEnvironmentVar("OV_BLOB_PATH");
    if(vpu_status == true) {
      LOGS_DEFAULT(INFO) << log_tag << "Importing the pre-compiled blob for this model which already exists in the directory 'ov_compiled_blobs'";
      exe_network_ = global_context_.ie_core.ImportModel(model_blob_path, hw_target, subgraph_context_.subgraph_name);
    } else {
      LOGS_DEFAULT(INFO) << log_tag << "Importing the pre-compiled blob from the path set by the user";
      if (compiled_blob_path.empty())
        throw std::runtime_error("The compiled blob path is not set");
      exe_network_ = global_context_.ie_core.ImportModel(compiled_blob_path, hw_target, subgraph_context_.subgraph_name);
    }
    import_blob_status = true;
    LOGS_DEFAULT(INFO) << log_tag << "Succesfully Created an executable network from a previously exported network";
  }

  if ((global_context_.use_compiled_network == true && import_blob_status == false) || vpu_status == false) {
    if(!openvino_ep::backend_utils::UseCompiledNetwork()) {
      ie_cnn_network_ = CreateCNNNetwork(model_proto, global_context_, subgraph_context_, const_outputs_map_);
      SetIODefs(model_proto, ie_cnn_network_, subgraph_context_.output_names, const_outputs_map_, global_context_.device_type);
      
      if (ValidateSubgraph(const_outputs_map_))
      return;
      
      OVConfig config;
      PopulateConfigValue(config);
        
      exe_network_ = global_context_.ie_core.LoadNetwork(ie_cnn_network_, hw_target, config, subgraph_context_.subgraph_name);
      LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
      
      if(global_context_.use_compiled_network && hw_target == "MYRIAD") {
        LOGS_DEFAULT(INFO) << log_tag << "Dumping the compiled blob for this model into the directory 'ov_compiled_blobs'";
        std::ofstream compiled_blob_dump{ov_compiled_blobs_dir + "/" + model_blob_name};
        exe_network_.Get().Export(compiled_blob_dump);
      }
    }
  }
#else

  if(hw_target == "MYRIAD") 
    vpu_status = true;
  if (!ImportBlob(hw_target, vpu_status)) {
  
  #if defined (OPENVINO_2021_4) 
    ie_cnn_network_ = CreateCNNNetwork(model_proto, global_context_, subgraph_context_, const_outputs_map_);
    SetIODefs(model_proto, ie_cnn_network_, subgraph_context_.output_names, const_outputs_map_, global_context_.device_type);  
  #else
    ie_cnn_network_ = CreateOVModel(model_proto, global_context_, subgraph_context_, const_outputs_map_);
  #endif 

  if (ValidateSubgraph(const_outputs_map_))
    return;
  
  // OV Config
  OVConfig config;
  PopulateConfigValue(config);
 
  //Enable caching
  EnableCaching();
  
  //Setting OpenCL queue throttling for GPU
  EnableGPUThrottling(config);

  #if defined(IO_BUFFER_ENABLED)
    if ((global_context.device_type.find("GPU") != std::string::npos)  && 
      (global_context_.context != nullptr) && 
      (openvino_ep::BackendManager::GetGlobalContext().is_wholly_supported_graph)) {
      LOGS_DEFAULT(INFO) << log_tag << "IO Buffering Enabled";    
      cl_context ctx = static_cast<cl_context>(global_context_.context); 
      remote_context_ = InferenceEngine::gpu::make_shared_context(global_context_.ie_core, "GPU", ctx);
      exe_network_ = global_context_.ie_core.LoadNetwork(ie_cnn_network_, remote_context_);
    } else {
    exe_network_ = global_context_.ie_core.LoadNetwork(ie_cnn_network_, hw_target, config, subgraph_context_.subgraph_name);
  }
  #else 
    exe_network_ = global_context_.ie_core.LoadNetwork(ie_cnn_network_, hw_target, config, subgraph_context_.subgraph_name);
  #endif
  LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
  }

#endif  
  //The infer_requests_ pool will be intialized with a default value of 8 infer_request's
  //The nireq value can also be configured to any num_of_threads during runtime
  size_t nireq = global_context_.num_of_threads;
  LOGS_DEFAULT(INFO) << log_tag << "The value of nireq being used is: " << nireq;
#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "The value of nireq being used is: " << nireq << std::endl;
  }
#endif
  inferRequestsQueue_ = std::unique_ptr<InferRequestsQueue>(new InferRequestsQueue(exe_network_, nireq));
}

bool BasicBackend::ValidateSubgraph(std::map<std::string, std::shared_ptr<ngraph::Node>>& const_outputs_map) {
  if (const_outputs_map.size() == subgraph_context_.output_names.size())
    subgraph_context_.is_constant = true;
  if (subgraph_context_.is_constant) {
    LOGS_DEFAULT(INFO) << log_tag << "The subgraph is a const. Directly moving to Infer stage.";
    return true;
  }
  return false;
}

void BasicBackend::PopulateCompiledDirectory(std::string hw_target, std::string& ov_compiled_blobs_dir, std::string& model_blob_name, bool& vpu_status) {
  std::ifstream blob_path;
  if(hw_target == "MYRIAD" && global_context_.use_compiled_network == true) {
    if(!openvino_ep::backend_utils::UseCompiledNetwork()) {
      std::size_t model_index = global_context_.onnx_model_path_name.find_last_of("/\\");
      std::string model_name= global_context_.onnx_model_path_name.substr(model_index+1);
      std::size_t model_extension_index = model_name.find_last_of(".");
      if(openvino_ep::BackendManager::GetGlobalContext().is_wholly_supported_graph) {
          model_blob_name = global_context_.onnx_model_name + "_" + "op_v_" + std::to_string(global_context_.onnx_opset_version) + "_" + model_name.substr(0,model_extension_index) + "_" + hw_target + "_" + subgraph_context_.subgraph_name + "_ov_" + "fully" + ".blob";
      }
      else {
          model_blob_name = global_context_.onnx_model_name + "_" + "op_v_" + std::to_string(global_context_.onnx_opset_version) + "_" + model_name.substr(0,model_extension_index) + "_" + hw_target + "_" + subgraph_context_.subgraph_name + "_ov_" + "partially" + ".blob";
      }
      if(global_context_.blob_dump_path == "" || global_context_.blob_dump_path == "\"" || global_context_.blob_dump_path.empty()) {
        ov_compiled_blobs_dir = openvino_ep::backend_utils::GetCurrentWorkingDir() + "/ov_compiled_blobs/";
      } else {
        ov_compiled_blobs_dir = global_context_.blob_dump_path + "/ov_compiled_blobs";
      }
      if(openvino_ep::backend_utils::IsDirExists(ov_compiled_blobs_dir)) {
        LOGS_DEFAULT(INFO) << log_tag << "'ov_compiled_blobs' directory already exists at the executable path";
      }
      else {
        CreateDirectory(ov_compiled_blobs_dir);
      }
      blob_path.open(ov_compiled_blobs_dir + "/" + model_blob_name);
      if (!blob_path.is_open()) {
          LOGS_DEFAULT(INFO) << log_tag << "Device specific Compiled blob doesn't exist for this model";
      } else {
          LOGS_DEFAULT(INFO) << log_tag << "Device specific Compiled blob already exists for this model";
          vpu_status = true;
      }
    }
  }
}

bool BasicBackend::ImportBlob(std::string hw_target, bool vpu_status) {
  
  const std::string compiled_blob_path = onnxruntime::GetEnvironmentVar("OV_BLOB_PATH");
  if (vpu_status == true && openvino_ep::backend_utils::UseCompiledNetwork() && !compiled_blob_path.empty() &&
    openvino_ep::BackendManager::GetGlobalContext().is_wholly_supported_graph) {
    LOGS_DEFAULT(INFO) << log_tag << "Importing the pre-compiled blob from the path set by the user";
    exe_network_ = global_context_.ie_core.ImportModel(compiled_blob_path, hw_target, subgraph_context_.subgraph_name);
    LOGS_DEFAULT(INFO) << log_tag << "Succesfully Created an executable network from a previously exported network";
    return true;
  } else {
    return false;
  }
}  

void BasicBackend::PopulateConfigValue(OVConfig& config) {
  #ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      config["PERF_COUNT"] = CONFIG_VALUE(YES);
    }
  #endif
    if (global_context_.device_type.find("MYRIAD") != std::string::npos) {
      if (subgraph_context_.set_vpu_config) {
        config["MYRIAD_DETECT_NETWORK_BATCH"] = CONFIG_VALUE(NO);
      }
    if (global_context_.enable_vpu_fast_compile) {
      config["MYRIAD_HW_INJECT_STAGES"] = CONFIG_VALUE(NO);
      config["MYRIAD_COPY_OPTIMIZATION"] = CONFIG_VALUE(NO);
    }
    //to check preprocessing inside model
    #if defined (OPENVINO_2021_4) || (OPENVINO_2022_1)
      config["MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL"] = CONFIG_VALUE(NO);
    #endif  
  }
}

void BasicBackend::EnableCaching() {
  if (global_context_.use_compiled_network == true) {
    std::string cache_dir_path;
    if (global_context_.blob_dump_path.empty()) {
      cache_dir_path = "ov_compiled_blobs";
    } else {
      cache_dir_path = global_context_.blob_dump_path;
    }
    LOGS_DEFAULT(INFO) << log_tag << "Enables Caching";
    global_context_.ie_core.SetCache(cache_dir_path);
  }
}

#if defined (OPENVINO_2022_1)
void BasicBackend::EnableGPUThrottling(OVConfig& config) {
  if (global_context_.enable_opencl_throttling == true && global_context_.device_type.find("GPU") != std::string::npos) {
    LOGS_DEFAULT(INFO) << log_tag << "Enabled OpenCL queue throttling for GPU device";
    config[GPU_CONFIG_KEY(PLUGIN_THROTTLE)] = "1";
  }
}
#endif

// Starts an asynchronous inference request for data in slice indexed by batch_slice_idx on
// an Infer Request indexed by infer_req_idx
void BasicBackend::StartAsyncInference(Ort::CustomOpApi& ort, OrtKernelContext* context, OVInferRequestPtr infer_request) {
  #if defined (OPENVINO_2022_1)
  auto graph_input_info = exe_network_.Get().inputs();
  for (auto input_info_iter = graph_input_info.begin();
    input_info_iter != graph_input_info.end(); ++input_info_iter) {
    auto input_names = input_info_iter->get_names(); 
    std::string input_name;
    for(auto it = input_names.begin(); it != input_names.end(); it++) {
      // Assign the input_name for the model
      std::string inp_name = *it;
      if (inp_name.find("graph_input_cast_") != std::string::npos) {
        continue;
      } else {
        input_name = *it;
      }
    }
    OVTensorPtr graph_input_blob; 
    graph_input_blob = infer_request->GetTensor(input_name);
    size_t batch_slice_idx = 0;
    FillInputBlob(graph_input_blob, batch_slice_idx, input_name, ort, context, subgraph_context_);
  }        
  #else
  auto graph_input_info = exe_network_.Get().GetInputsInfo();
  for (auto input_info_iter = graph_input_info.begin();
       input_info_iter != graph_input_info.end(); ++input_info_iter) {
    // Get OpenVINO's input buffer
    OVTensorPtr graph_input_blob;
    std::string input_name = input_info_iter->first;
    graph_input_blob = infer_request->GetTensor(input_name);
    auto precision = input_info_iter->second->getPrecision();
    size_t batch_slice = 0;
    FillInputBlob(graph_input_blob, batch_slice, input_name, ort, context, precision, subgraph_context_);
  }
  #endif
  // Start Async inference
  infer_request->StartAsync();
}

#ifdef IO_BUFFER_ENABLED
//Wait for Remote Aynchronous inference completion
void BasicBackend::StartRemoteAsyncInference(Ort::CustomOpApi& ort, OrtKernelContext* context, OVInferRequestPtr infer_request) {
  
  #if defined (OPENVINO_2022_1)
  auto graph_input_info = exe_network_.get().inputs();
  for (auto input_info_iter = graph_input_info.begin();
    input_info_iter != graph_input_info.end(); ++input_info_iter) {
    auto input_names = input_info_iter->get_names(); 
    std::string input_name;
    for(auto it = input_names.begin(); it != input_names.end(); it++) {
      // Assign the input_name for the model
      std::string inp_name = *it;
      if (inp_name.find("graph_input_cast_") != std::string::npos) {
        continue;
      } else {
        input_name = *it;
      }
    }
  #else 
  auto graph_input_info = exe_network_.get().GetInputsInfo();
  for (auto input_info_iter = graph_input_info.begin();
       input_info_iter != graph_input_info.end(); ++input_info_iter) {

    std::string input_name = input_info_iter->first;
  #endif   
    // Kernel Context Input Buffer
    const OrtValue* tensor = ort.KernelContext_GetInput(context, subgraph_context_.input_names.at(input_name));
    // If the ORTValue wraps a device pointer
    auto mem_info = ort.GetTensorMemoryInfo(tensor);
    if (strcmp(mem_info->name, OpenVINO_GPU) == 0) {
      //Get the shared buffer pointer
      const void *tensor_data = ort.GetTensorData<void *>(tensor);
      const cl::Buffer* shared_buffer_const = static_cast<const cl::Buffer*>(tensor_data);
      cl::Buffer* shared_buffer = const_cast<cl::Buffer *>(shared_buffer_const);
      //Create an Input Remote Blob
      OVTensorPtr graph_input_blob = InferenceEngine::gpu::make_shared_blob(input_info_iter->second->getTensorDesc(), remote_context_, *shared_buffer);
      infer_request->SetTensor(input_name, graph_input_blob);
      } else {
        OVTensorPtr graph_input_blob; 
        graph_input_blob = infer_request->GetTensor(input_name);
        size_t batch_slice_idx = 0;
        #if defined (OPENVINO_2022_1)
        FillInputBlob(graph_input_blob, batch_slice_idx, input_name, ort, context, subgraph_context_);
        #else 
        auto precision = input_info_iter->second->getPrecision();
        FillInputBlob(graph_input_blob, batch_slice_idx, input_name, ort, context, precision, subgraph_context_);
      }
    }
  }

  //Set the output blob as remote blob
  #if defined (OPENVINO_2022_1)
  auto graph_output_info = exe_network_.get().outputs();
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter) {
    auto output_names = output_info_iter->get_names();
    std::string output_name;
    for(auto it = output_names.begin(); it != output_names.end(); it++) {
      // Assign the output_name for the model
      std::string out_name = *it;
      if (out_name.find("graph_output_cast_") != std::string::npos) {
        continue;
      } else {
        output_name = *it;
      }
    }
  #else
  auto graph_output_info = exe_network_.GetOutputsInfo();
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter) {
    // Get Ort Output Tensor
    auto output_name = output_info_iter->first;
  #endif 

    size_t batch_size = 1;
    auto tensor = GetOutputTensor(ort, context, batch_size, infer_request, output_name, subgraph_context_.output_names);
    auto mem_info = ort.GetTensorMemoryInfo(tensor);
    // Check if ORT Value wraps a device pointer
    if (strcmp(mem_info->name, OpenVINO_GPU) == 0) {
      const void *tensor_data = ort.GetTensorData<void *>(tensor);
      const cl::Buffer* shared_buffer_const = static_cast<const cl::Buffer*>(tensor_data);
      cl::Buffer* shared_buffer = const_cast<cl::Buffer *>(shared_buffer_const);
      // Create a shared Blob, set the Infer Request Output Blob
        OVTensorPtr graph_output_blob = InferenceEngine::gpu::make_shared_blob(output_info_iter->second->getTensorDesc(), remote_context_, *shared_buffer);
        infer_request->set_blob(output_name, graph_output_blob);
    }
  }
  #endif 
  // Start Async inference
  infer_request->StartAsync();
 
}
#endif

// Wait for asynchronous inference completion on an Infer Request object indexed by infer_req_idx
// and copy the results into a slice location within the batched output buffer indexed by batch_slice_idx
void BasicBackend::CompleteAsyncInference(Ort::CustomOpApi& ort, OrtKernelContext* context, OVInferRequestPtr infer_request) {
  // Wait for Async inference completion

  infer_request->Wait();
  #if defined (OPENVINO_2022_1)
  auto graph_output_info = exe_network_.Get().outputs();
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter) {
    OVTensorPtr graph_output_blob;
    auto output_names = output_info_iter->get_names();
    std::string output_name;
    for(auto it = output_names.begin(); it != output_names.end(); it++) {
      // Assign the output_name for the model
      std::string out_name = *it;
      if (out_name.find("graph_output_cast_") != std::string::npos) {
        continue;
      } else {
        output_name = *it;
      }
    }
    graph_output_blob = infer_request->GetTensor(output_name);
    size_t batch_size = 1;
    auto output_tensor = GetOutputTensor(ort, context, batch_size, infer_request, output_name, subgraph_context_.output_names);
    auto mem_info = ort.GetTensorMemoryInfo(output_tensor);
    if (strcmp(mem_info->name, OpenVINO_GPU) == 0) {
      return;
    } else {
      size_t batch_slice = 0;
      FillOutputBlob(graph_output_blob, output_tensor, ort, batch_slice);
    }
  }
  #else
  auto graph_output_info = exe_network_.Get().GetOutputsInfo();
  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter) {
    // Get OpenVINO's output blob
    OVTensorPtr graph_output_blob;
    auto output_name = output_info_iter->first;
    graph_output_blob = infer_request->GetTensor(output_name);
    size_t batch_size = 1;
    auto output_tensor = GetOutputTensor(ort, context, batch_size, infer_request, output_name, subgraph_context_.output_names);
    auto mem_info = ort.GetTensorMemoryInfo(output_tensor);
    if (strcmp(mem_info->name, OpenVINO_GPU) == 0) {
      return;
    } else {
      auto precision = output_info_iter->second->getPrecision();
      size_t batch_slice = 0;
      FillOutputBlob(graph_output_blob, output_tensor, ort, precision, batch_slice);
    }
  }
  #endif
  
  if (!const_outputs_map_.empty()) {
    for (auto item : const_outputs_map_) {
      auto out_name = item.first;
      auto node = item.second;
      auto output_tensor = GetOutputTensor(ort, context, out_name, subgraph_context_.output_names, node);
      auto mem_info = ort.GetTensorMemoryInfo(output_tensor);
      if (strcmp(mem_info->name, OpenVINO_GPU) == 0) {
        ORT_THROW(log_tag + "IO Buffering is not supported for constant subgraphs");
      } else {
        FillOutputsWithConstantData(ort, node, output_tensor);
      }
    }  
  }
}

void BasicBackend::Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  // Preliminary Thread safety mechanism
  // currently allows a maximum of 8 Infer request's to paralelly execute at the same time

  LOGS_DEFAULT(INFO) << log_tag << "Running graph " << subgraph_context_.subgraph_name;
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";

  if (subgraph_context_.is_constant) {
    for (auto item : const_outputs_map_) {
      auto out_name = item.first;
      auto node = item.second;
      auto output_tensor = GetOutputTensor(ort, context, out_name, subgraph_context_.output_names, node);
      FillOutputsWithConstantData(ort, node, output_tensor);
    }
    // Get Output tensors
    LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
    //Enable CI Logs
    if(IsCILogEnabled()) {
      std::cout << "Inference successful" << std::endl;
    }

  } else {
      //Requesting for an idle infer_request from a pool of infer_requests_
      OVInferRequestPtr infer_request;
      infer_request = inferRequestsQueue_->getIdleRequest();
	   
      #ifdef IO_BUFFER_ENABLED
      if ((global_context_.device_type.find("GPU") != std::string::npos)  && 
          (global_context_.context != nullptr) && 
          (openvino_ep::BackendManager::GetGlobalContext().is_wholly_supported_graph)) {
        StartRemoteAsyncInference(ort, context, infer_request);
      } else {
        StartAsyncInference(ort, context, infer_request);
      }
      #else 
        StartAsyncInference(ort, context, infer_request);
      #endif 

      CompleteAsyncInference(ort, context, infer_request);
  
      // Get Output tensors
      LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
      //Enable CI Logs
      if (IsCILogEnabled()) {
        std::cout << "Inference successful" << std::endl;
      }

      //Once the inference is completed, the infer_request becomes free and is placed back into pool of infer_requests_
      inferRequestsQueue_->putIdleRequest(infer_request);
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      inferRequestsQueue_->printstatus();  //Printing the elements of infer_requests_ vector pool only in debug mode
      std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
      printPerformanceCounts(infer_request, std::cout, hw_target);
    }
#endif
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime
