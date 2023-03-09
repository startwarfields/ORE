/**
* Project to test out ONNX Runtime
*/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "src/ml.h"
#include <ctime>
#include <chrono>
using namespace std::chrono;

void test_inference_fast()
{
  ML ml;
  ml.initialize_onnx();
  int i = 0;
  while (i < 100000) {
//    ml.initialize_onnx();
  ml.inference_onnx();
  i++;
  
  }
}

void test_inference_slow()
{ 
  ML ml;
  int i = 0;
  while (i < 100000) {
//    ml.initialize_onnx();
    //
  ml.initialize_onnx();
  ml.inference_onnx();
  i++;
  }
}



int main(int argc, char* argv[]) {

  std::cout << "Hello!" << "\n";
  // Seed the RNG
  srand(static_cast<unsigned> (time(0)));

  auto start = high_resolution_clock::now();

  // Fast only loads the model once, then inferences separately
  test_inference_fast();
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<seconds>(stop - start);

  std::cout <<"Fast Version took: " << duration.count() << "\n";

  // Slow loads the model every inference
  //
  
  start = high_resolution_clock::now();
  test_inference_slow();
  stop = high_resolution_clock::now();
  duration = duration_cast<seconds>(stop - start);

  std::cout <<"Slow Version took: " << duration.count() << "\n";


}





