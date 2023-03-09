#include <stdlib.h>
#include <memory>
#include <onnxruntime_cxx_api.h>

class ML
{

  public:
    
    /*
     * Initialize ONNX Runtime for ML
     * Note: This has high cost 
   */
    void initialize_onnx(void);

    /*
     * Inference ONNX Runtime
     * TODO: Change Function Signature 
     */
    void inference_onnx(void);


    std::vector<float> generate_random_floats(int num_of_floats=4);
  private:
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;

};
