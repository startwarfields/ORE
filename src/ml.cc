#include "ml.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <cstdint>
#include <iostream>
#include <ctime>

using std::make_unique;
using std::int64_t;
using std::cout;

typedef int64_t int64;

// ML Code goes here
//

void ML::initialize_onnx() {

  // TODO: Refactor to separate object.
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  env_ = make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");

  session_ = make_unique<Ort::Session>(*env_, "/home/startwarfields/Github/macsim/model.onnx", opts);

}


void ML::inference_onnx() {
  Ort::RunOptions run_options;
  run_options.SetRunLogVerbosityLevel(0);
//  run_options.SetRunLogVerbosityLevel(4);
            
  const char* input_names[] = {"input"};
  const char* output_names[] = {"output_label", "output_probability"};
  std::vector<float> input_vec = generate_random_floats(4);
//  std::vector<std::vector<//float>> output_vec(2,3); // resize the vector to hold 3 values
  std::vector<int64_t> input_shape = {1, 4}; // use int64_t instead of int

  auto memory_info_ = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto default_allocator = std::make_unique<Ort::AllocatorWithDefaultOptions>();
  const int64_t input_size = input_vec.size();
 // const int64_t output_size = output_vec.size();

  auto input_tensor_ = Ort::Value::CreateTensor<float>(memory_info_, input_vec.data(), input_size,
                                                       input_shape.data(), input_shape.size());
  //auto output_tensor_ = Ort::Value::CreateTensor<float>(memory_info_, output_vec.data(), output_size,
//                                                        output_shape.data(), output_shape.size());
  

  auto output_tensors = session_->Run(run_options, input_names, &input_tensor_, 1,
            output_names, 2);
  int64 name = *output_tensors[0].GetTensorMutableData<int64>();
  auto outputInfo = output_tensors[0].GetTensorTypeAndShapeInfo();
  //cout << "GetElementType: " << outputInfo.GetElementType() << "\n";
  //cout << "Dimensions of the output: " << outputInfo.GetShape().size() << "\n";
  //cout << "Shape of the output: ";
//  for (unsigned int shapeI = 0; shapeI < outputInfo.GetShape().size(); shapeI++)
  //  std::cout << outputInfo.GetShape()[shapeI] << ", ";

//  cout << name << "\n";
  auto memap = output_tensors[1].GetValue(0, *default_allocator.get());
//  cout << "============Predictions==========" << "\n";

  auto keys = memap.GetValue(0, *default_allocator.get());
  

  auto values = memap.GetValue(1, *default_allocator.get());

  //for (int i = 0; i < 3; i++)
  //{
    //int64 i_key = keys.GetTensorMutableData<int64>()[i];
    //float i_value = values.GetTensorMutableData<float>()[i];
  //  std::cout << "Class: " << i_key << " Prediction: " << i_value << "\n"; 
//  }
 
//  printf("%f", *my_float); 
}

// This will generate random floats between 0.0 and 5.0
std::vector<float> ML::generate_random_floats(int num_floats) {

 
  
  std::vector<float> random_floats;
  while(num_floats > 0)
  {
    float r2 = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/5.0));
    random_floats.push_back(r2);
    num_floats--;
  }
  return random_floats;
}
