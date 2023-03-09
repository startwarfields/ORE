/**
* Project to test out ONNX Runtime
*/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "src/ml.h"

int main(int argc, char* argv[]) {

  std::cout << "Hello!" << std::endl;

  ML ml; 
  ml.initialize_onnx();
  ml.inference_onnx();

}

