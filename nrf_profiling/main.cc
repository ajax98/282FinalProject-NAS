/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the sp`ecific language governing permissions and
limitations under the License.
==============================================================================*/
#include <string.h>
#include <iostream>
// #include "tensorflow/lite/micro/examples/person_detection/main_functions.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/micro/kernels/mem_logger.h"
// #include "tensorflow/lite/micro/allocation_logger.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <cstdlib>
#include <vector> 
#include <typeinfo>
#include "types.h"
#include "conv.h"
#include "math.h"
#include "depthwiseconv_float.h"


extern "C" {
#include "virtual_timer.h"
#include "app_error.h"
#include "nrf.h"
#include "nrf_delay.h"
#include "nrfx_gpiote.h"
#include "nrf_gpio.h"
#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"
#include "nrf_pwr_mgmt.h"
#include "nrf_serial.h"

#include "buckler.h"
}


typedef std::vector<std::vector<std::vector<std::vector<float>>>> tensor_t;

template<typename T>                                                      //default argument
std::vector<std::vector<std::vector<std::vector<T>>>> make_4d_vector(int dim1, int dim2, int dim3, int dim4, T value = T{})
{
    return std::vector<std::vector<std::vector<std::vector<T>>>>(dim1, std::vector<std::vector<std::vector<T>>>(dim2, std::vector<std::vector<T>>(dim3, std::vector<T>(dim4, value))));
}


 void print_tensor(tensor_t& vec)
 {
    for (int i = 0; i < vec.size(); i++)
    {
      for (int j = 0; j < vec[i].size(); j++)
      {
        for (int k = 0; k < vec[i][j].size(); k++)
        {
          for (int l = 0; l < vec[i][j][k].size(); l++){

             std::cout << vec[i][j][k][l] << " ";
          }
            std::cout << "\n";
        }
      }
    }
 }

 void print_special_tensor(tensor_t& vec)
 {
    for (int i = 0; i < vec.size(); i++)
    {
      for (int j = 0; j < vec[i].size(); j++)
      {
        for (int k = 0; k < vec[i][j].size(); k++)
        {
          // for (int l = 0; l < vec[i][j][k].size(); l++){

          //    std::cout << vec[i][j][k][l] << " ";
          // }
          std::cout << vec[i][j][k][0] << " ";

          
        }
        std::cout << "\n";
      }
    }
 }

  void print_vector(std::vector<float>& vector, int size)
 {
    for(int i = 0; i < size; i++){
      printf("I: %d Value: %f \n", i, vector[i]);
      // std::cout << i << ": " << vector[i] << "\n";
    }
 }

 void run_convolution(std::vector<int32_t>& input_shape, std::vector<int32_t>&  output_shape, float * input_array, float * output_array, int kernel, int stride)
 {



    // std::vector<float> filter_array = std::vector<float> (output_shape[3]*kernel*kernel*input_shape[3], 2.0);
    float * filter_array = (float*)calloc(48*kernel*kernel*48, sizeof(float));
    std::vector<int32_t> filter_shape{output_shape[3],kernel,kernel,input_shape[3]};

    tflite::RuntimeShape input_r_shape = tflite::RuntimeShape(input_shape.size(), input_shape.data());
    tflite::RuntimeShape output_r_shape = tflite::RuntimeShape(output_shape.size(), output_shape.data());
    tflite::RuntimeShape filter_r_shape = tflite::RuntimeShape(filter_shape.size(), filter_shape.data());

    tflite::ConvParams op_params;
    op_params.padding_type = tflite::PaddingType::kSame;
    op_params.padding_values.width = (kernel-1)/2;
    op_params.padding_values.height = (kernel-1)/2;
    op_params.stride_width = stride;
    op_params.stride_height = stride;
    op_params.dilation_width_factor = 1;
    op_params.dilation_height_factor = 1;
    op_params.float_activation_min = std::numeric_limits<float>::min();
    op_params.float_activation_max = std::numeric_limits<float>::max();


    tflite::reference_ops::Conv(op_params, input_r_shape, (float *) input_array, 
                              filter_r_shape, (float *) filter_array,
                               tflite::RuntimeShape(), NULL,
                              output_r_shape, (float *) output_array,
                              tflite::RuntimeShape(), NULL);

    free(filter_array);


}

 void run_depthwise_convolution(std::vector<int32_t>& input_shape, std::vector<int32_t>&  output_shape, float * input_array, float * output_array, int kernel, int stride)
 {



    // std::vector<float> filter_array = std::vector<float> (output_shape[3]*kernel*kernel*input_shape[3], 2.0);
    float * filter_array = (float*)calloc(kernel*kernel*output_shape[3], sizeof(float));
    std::vector<int32_t> filter_shape{output_shape[3],kernel,kernel,input_shape[3]};

    tflite::RuntimeShape input_r_shape = tflite::RuntimeShape(input_shape.size(), input_shape.data());
    tflite::RuntimeShape output_r_shape = tflite::RuntimeShape(output_shape.size(), output_shape.data());
    tflite::RuntimeShape filter_r_shape = tflite::RuntimeShape(filter_shape.size(), filter_shape.data());

    tflite::DepthwiseParams op_params;
    // Padding type is ignored, but still set.
    op_params.padding_type = tflite::PaddingType::kSame;
    op_params.padding_values.width = (kernel-1)/2;
    op_params.padding_values.height = (kernel-1)/2;
    op_params.stride_width = stride;
    op_params.stride_height = stride;
    op_params.dilation_width_factor = 1;
    op_params.dilation_height_factor = 1;
    op_params.depth_multiplier = 1;
    op_params.float_activation_min = std::numeric_limits<float>::min();
    op_params.float_activation_max = std::numeric_limits<float>::max();

    tflite::reference_ops::DepthwiseConv(op_params, input_r_shape, (float *) input_array, 
                              filter_r_shape, (float *) filter_array,
                               tflite::RuntimeShape(), NULL,
                              output_r_shape, (float *) output_array);

    free(filter_array);


}

// #include "convolution.h"

//COMPILE COMMAND
// g++ -o convTest -std=c++11 main.cc convolution.cc -I .
// #include "nrf.h"
// #include "nrf_delay.h"
// #include "nrf_gpio.h"

// Pin definitions
// #define LED NRF_GPIO_PIN_MAP(0,14)

// namespace MemLogger{
//  Event g_events[BUFFER_SIZE];
//  int index=-1;
// }

// namespace AllocationLogger{
//  AllocationInfo allocations[BUFFER_SIZE];
//  int index=-1;
// }
// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
int main(int argc, char* argv[]) {

  ret_code_t error_code = NRF_SUCCESS;

  // initialize RTT library
  error_code = NRF_LOG_INIT(NULL);
  APP_ERROR_CHECK(error_code);
  NRF_LOG_DEFAULT_BACKENDS_INIT();
  // tflite::ErrorReporter* error_reporter = nullptr;
  // static tflite::MicroErrorReporter micro_error_reporter;
  // error_reporter = &micro_error_reporter;
  // error_reporter->Report("WER");
  // // memset((void *)0x200008f0, 0xaa, 0x34e7c);
  // setup();
  // nrf_gpio_cfg_output(LED);
  // nrf_gpio_pin_toggle(LED);
  // nrf_delay_ms(500);
  // nrf_gpio_pin_toggle(LED);
  // // nrf_gpio_pin_toggle(LED);
  // nrf_delay_ms(500);
  // nrf_gpio_pin_toggle(LED);
  // std::cout << "Entering Convolution! " << "\n";
  // int input_arr [10][10][3];
  // int output_arr [5][5][6];

  // for (int row = 0; row < 10; row++){
  //   for (int col = 0; col < 10; col++){
  //     for (int depth = 0; depth < 3; depth++){
  //       input_arr[row][col][depth] = (rand() % 1000) + 1;
  //     }
  //   }   
  // }
  // std::cout << "Input arr: \n" << input_arr << "\n";

  // std::cout << "Creating vector! " << "\n";

  // std::vector <float> vecOfInts(5);

  // std::vector < std::vector<int>> vecOfVecs(5);

  // float * x;
  // x = vecOfInts.data();

  // std::cout << "X at 0: " << x[0] << "\n";

  // std::vector<std::vector<std::vector<float>>> input_array = std::vector<std::vector<std::vector<float>>>(10, std::vector<std::vector<float>>(10, std::vector<float>(3, 0)));

  // tensor_t input_array = make_4d_vector(1,3,3,1,((float)2.0));
  // tensor_t filter_array = make_4d_vector(1,3,3,1,((float)2.0));
  // tensor_t output_array = make_4d_vector(1,3,3,1,((float)2.0));


  // std::vector<float> input_array = std::vector<float> (3*3*3, 2.0);
  // std::vector<float> output_array = std::vector<float> (3*3*3, 2.0);
  // std::vector<float> filter_array = std::vector<float> (3*3*3, 2.0);


  // std::vector<int32_t> input_shape{1,3,3,1};
  // std::vector<int32_t> filter_shape{1,3,3,1};
  // std::vector<int32_t> output_shape{1,3,3,1};

  // std::cout << "Start convolution!" << "\n";
 
  // run_convolution(input_shape, output_shape, input_array, output_array, 3, 1);

  // std::cout << "End convolution!" << "\n";





  // int input_height = 14;
  // int input_width = 14;
  // int input_depth = 3;
  // std::vector<float> input_array = std::vector<float> (input_height*input_width*input_depth, 2.0);
  // std::vector<int32_t> input_shape{1,input_height,input_width,input_depth};

  // int output_height = 14;
  // int output_width = 14;
  // int output_depth = 3;
  // std::vector<float> output_array = std::vector<float> (output_height*output_width*output_depth, 2.0);
  // std::vector<int32_t> output_shape{1,output_height,output_width,output_depth};

  // int stride = 1;
  // int expansion = 1;
  // int group = 1;

  // int intermediate_1_height = input_height;
  // int intermediate_1_width = input_width;
  // int intermediate_1_depth = expansion * input_depth;
  // std::vector<float> int_1_array = std::vector<float> (intermediate_1_height*intermediate_1_width*intermediate_1_depth, 2.0);
  // std::vector<int32_t> intermediate_1_shape{1,intermediate_1_height,intermediate_1_width,intermediate_1_depth};


  // int intermediate_2_height = input_height/stride;
  // int intermediate_2_width = input_width/stride;
  // int intermediate_2_depth = intermediate_1_depth;
  // std::vector<float> int_2_array = std::vector<float> (intermediate_2_height*intermediate_2_width*intermediate_2_depth, 2.0);
  // std::vector<int32_t> intermediate_2_shape{1,intermediate_2_height,intermediate_2_width,intermediate_2_depth};

  // // print_vector(output_array, 14*14*3);
  // printf("Starting convolution!\n");
  // volatile uint32_t start_time = read_timer();

  // for (int i = 0; i < 1000; i++){
  //   run_convolution(input_shape, intermediate_1_shape, input_array, int_1_array, 1, 1);
  //   run_convolution(intermediate_1_shape, intermediate_2_shape, int_1_array, int_2_array, 3, stride);
  //   run_convolution(intermediate_2_shape, output_shape, int_2_array, output_array, 1, 1);
  // }
  // volatile uint32_t end_time = read_timer();
  // volatile uint32_t time_taken = end_time-start_time;

  // printf("End convolution!\n");
  // printf("Cycles taken:  %d \n", time_taken);
  // print_vector(output_array, 14*14*3);

  //Layer info
  std::vector<int> strides{1,2,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
  std::vector<int> depths{8,12,12,12,12,16,16,16,16,32,32,32,32,56,56,56,56,92,92,92,92,92};

  //Block info
  std::vector<int> expansions{6,1,1,3,6,1,1,3};
  std::vector<int> kernels{5,3,3,3,3,5,5,5};
  std::vector<int> groups{1,1,2,1,1,1,2,1};


  int input_height = 8;
  int input_width = 8;
  int input_depth = 8;
  int output_height = 0;
  int output_width = 0;
  int output_depth = 0;

  printf("PROFILE START \n");
  for (int layer = 0; layer < 22; layer++){
    // printf("-----------LAYER %d-----------\n", layer);
    int stride = strides[layer];
    int depth = depths[layer];
    // printf("DEPTH: %d", depth);
    // std::vector<float> input_array = std::vector<float> (input_height*input_width*input_depth, 2.0);
    float * input_array = (float*)calloc(input_height*input_width*input_depth, sizeof(float));

    std::vector<int32_t> input_shape{1,input_height,input_width,input_depth};
    
    int output_height = (input_height/stride);
    int output_width = (input_width/stride);
    int output_depth = depth;
    float * output_array = (float*)calloc(output_height*output_width*output_depth, sizeof(float));

    // std::vector<float> output_array = std::vector<float> (output_height*output_width*output_depth, 2.0);
    std::vector<int32_t> output_shape{1,output_height,output_width,output_depth};

    virtual_timer_init();

    for (int block = 0; block < 8; block++){

      int expansion = expansions[block];
      int kernel =  kernels[block];
      int group = groups[block];
      // printf("expansion: %d", expansion);
      // printf("kernel: %d", kernel);


      int intermediate_1_height = input_height;
      int intermediate_1_width = input_width;
      int intermediate_1_depth = expansion * input_depth;
      // std::vector<float> int_1_array = std::vector<float> (intermediate_1_height*intermediate_1_width*intermediate_1_depth, 2.0);
      float * int_1_array = (float*)calloc(intermediate_1_height*intermediate_1_width*intermediate_1_depth, sizeof(float));
      std::vector<int32_t> intermediate_1_shape{1,intermediate_1_height,intermediate_1_width,intermediate_1_depth};


      int intermediate_2_height = (input_height/stride);
      int intermediate_2_width = (input_width/stride);
      int intermediate_2_depth = intermediate_1_depth;
      float * int_2_array = (float*)calloc(intermediate_2_height*intermediate_2_width*intermediate_2_depth, sizeof(float));

      // std::vector<float> int_2_array = std::vector<float> (intermediate_2_height*intermediate_2_width*intermediate_2_depth, 2.0);
      std::vector<int32_t> intermediate_2_shape{1,intermediate_2_height,intermediate_2_width,intermediate_2_depth};

      // printf("Size of float: %d", sizeof(float));
      // printf("Worst case Memory Used: %d: ", intermediate_1_height*intermediate_1_width*intermediate_1_depth*4 + intermediate_2_height*intermediate_2_width*intermediate_2_depth*4 + input_height*input_width*input_depth * 4 + output_height*output_width*output_depth * 4 + intermediate_1_depth*5*5*4);
      volatile uint32_t start_time = read_timer();

      for (int i = 0; i < 10; i++){
        run_convolution(input_shape, intermediate_1_shape, input_array, int_1_array, 1, 1);
        run_depthwise_convolution(intermediate_1_shape, intermediate_2_shape, int_1_array, int_2_array, kernel, stride);
        run_convolution(intermediate_2_shape, output_shape, int_2_array, output_array, 1, 1);
      }

      volatile uint32_t end_time = read_timer();
      volatile uint32_t time_taken = end_time-start_time;

      free(int_1_array);
      free(int_2_array);
      printf("%f ", (float) 37.5*(time_taken/1000000.0));

      // printf("BLOCK %d Time: %f (s) \n", block, (float) (time_taken/1000000.0));

    }
    printf("\n");
    free(input_array);
    free(output_array);
    input_height = output_height;
    input_width = output_width;
    input_depth = output_depth;


  }

  // std::cout << "Input Shape at 1: " << input_r_shape.Dims(1) << "\n";

  // for (int i = 0; i < input_shape.size(); i++){
  //   std::cout << input_shape[i] << " "; 
  // }
  // std::cout << "Input Array: " << "\n";
  // print_vector(input_array, 9);
  // std::cout << "Output Array: " << "\n";
  // print_vector(input_array, 9);
  // std::cout << "Filter Array: " << "\n";
  // print_vector(input_array, 9);



  // std::cout << "Calling Function! " << "\n";
  // tflite::reference_ops::Conv(op_params, input_r_shape, (float *) input_array.data(), 
  //                             filter_r_shape, (float *) filter_array.data(),
  //                              tflite::RuntimeShape(), NULL,
  //                             output_r_shape, (float *) output_array.data(),
  //                             tflite::RuntimeShape(), NULL);

 

  printf("END OF MAIN! ");
  // for (int x: vecOfInts){
  //   std::cout << x << "\n";
  // }

  // std::cout << "vecOfVecs size: " << vecOfVecs.size() << " vecOfVec 2nd dim: " << vecOfVecs[0].size() << "\n";
  // // convolution(input_arr,output_arr,10,10);

  // for (int y = 0; y < vecOfVecs.size(); y++){
  //   // std::cout << "Vec of Vecs at " << typeid(vecOfVecs[y]).name()<< "\n";
  //   vecOfVecs[y].push_back(1);
  //   std::cout << "Vec of Vecs at " << y << vecOfVecs[y].size() << "\n";
  // }


  // while (true) {
  //   loop();
  //   nrf_gpio_pin_toggle(LED);
  //   nrf_delay_ms(500);
  //   nrf_gpio_pin_toggle(LED);

  // }
  return 0;
}
