# nRF application makefile
PROJECT_NAME = $(shell basename "$(realpath ./)")

# Configurations
NRF_IC = nrf52840
SDK_VERSION = 15
SOFTDEVICE_MODEL = s132


TENSORFLOW_ACTUALPATH=/home/ajay/Desktop/MastersProject/tensorflowlite/tensorflow
SDK_VARS += _HEAP_SIZE=0x35000
# __STACK_SIZE=0x25000


# Source and header files
APP_HEADER_PATHS += .
#APP_HEADER_PATHS += ./third_party/gemmlowp
#APP_HEADER_PATHS += ./third_party/flatbuffers/include

APP_SOURCE_PATHS += .
#APP_SOURCE_PATHS += ./tensorflow/lite/experimental/micro\
./tensorflow/lite/c\
./tensorflow/lite/core/api\
./tensorflow/lite/kernels\
./tensorflow/lite/experimental/micro/kernels\
./tensorflow/lite/kernels/internal/\

#TF_SOURCE_PATHS = /tensorflow/lite/micro/
#APP_SOURCE_PATHS += $(addprefix $(TENSORFLOW_ACTUALPATH), $(TF_SOURCE_PATHS))
#APP_SOURCE_PATHS += ./tensorflow/lite/micro/ ./tensorflow/lite/kernels/  ./tensorflow/lite/micro/kernels/ ./tensorflow/lite/micro/examples/hello_world/ ./tensorflow/lite/core/api/ ./tensorflow/lite/kernels/internal/ 

APP_SOURCES = $(notdir $(wildcard ./*.c))
APP_SOURCES += $(notdir $(wildcard ./*.cc))
#APP_SOURCES += ./tensorflow/lite/experimental/micro/micro_error_reporter.cc\
./tensorflow/lite/experimental/micro/micro_mutable_op_resolver.cc\
./tensorflow/lite/experimental/micro/simple_tensor_allocator.cc\
./tensorflow/lite/experimental/micro/debug_log.cc\
./tensorflow/lite/experimental/micro/debug_log_numbers.cc\
./tensorflow/lite/experimental/micro/micro_interpreter.cc\
./tensorflow/lite/experimental/micro/kernels/depthwise_conv.cc\
./tensorflow/lite/experimental/micro/kernels/softmax.cc\
./tensorflow/lite/experimental/micro/kernels/all_ops_resolver.cc\
./tensorflow/lite/experimental/micro/kernels/fully_connected.cc\
./tensorflow/lite/c/c_api_internal.c\
./tensorflow/lite/core/api/error_reporter.cc\
./tensorflow/lite/core/api/flatbuffer_conversions.cc\
./tensorflow/lite/core/api/op_resolver.cc\
./tensorflow/lite/kernels/kernel_util.cc\
./tensorflow/lite/kernels/internal/quantization_util.cc\
./tensorflow/lite/experimental/micro/simple_tensor_allocator_test.cc
#APP_SOURCES += ./tensorflow/lite/experimental/micro/micro_error_reporter.cc ./tensorflow/lite/experimental/micro/micro_mutable_op_resolver.cc ./tensorflow/lite/experimental/micro/simple_tensor_allocator.cc ./tensorflow/lite/experimental/micro/debug_log.cc ./tensorflow/lite/experimental/micro/debug_log_numbers.cc ./tensorflow/lite/experimental/micro/micro_interpreter.cc ./tensorflow/lite/experimental/micro/kernels/depthwise_conv.cc ./tensorflow/lite/experimental/micro/kernels/softmax.cc ./tensorflow/lite/experimental/micro/kernels/all_ops_resolver.cc ./tensorflow/lite/experimental/micro/kernels/fully_connected.cc ./tensorflow/lite/c/c_api_internal.c ./tensorflow/lite/core/api/error_reporter.cc ./tensorflow/lite/core/api/flatbuffer_conversions.cc ./tensorflow/lite/core/api/op_resolver.cc ./tensorflow/lite/kernels/kernel_util.cc ./tensorflow/lite/kernels/internal/quantization_util.cc  ./tensorflow/lite/experimental/micro/micro_error_reporter_test.cc

#TF_SOURCES = /tensorflow/lite/micro/micro_error_reporter.cc /tensorflow/lite/micro/micro_interpreter.cc /tensorflow/lite/micro/kernels/all_ops_resolver.cc 
#APP_SOURCES += $(addprefix $(TENSORFLOW_ACTUALPATH), $(TF_SOURCES)) 
APP_SOURCES =  ./main.cc ./virtual_timer.c
#CXXFLAGS += -O3 -DNDEBUG --std=c++11 -g -DTF_LITE_STATIC_MEMORY -fpermissive
#TF_LITE_USE_GLOBAL_ROUND=1

# Path to base of nRF52-base repo
NRF_BASE_DIR = ../../nrf52x-base/

# Include board Makefile (if any)
include ../../boards/buckler_revB/Board.mk

# Include main Makefile
include $(NRF_BASE_DIR)make/AppMakefile.mk
#@echo "BUILDDIR: $(BUILDDIR)"
#@echo "ELF: $(ELF)"
#@echo "CFLAGS: $(CFLAGS)"
#@echo "CXXFLAGS: $(CXXFLAGS)"
#@echo "OBJS: $(OBJS)" 
#@echo "VPATH: $(VPATH)"
#@echo "HEADER INCLUDES: $(HEADER_INCLUDES)"
list:
	@echo "APP SOURCES: $(APP_SOURCES)"
	@echo "APP SOURCES PATH: $(APP_SOURCE_PATHS)"	
	@echo "CXXFLAGS: $(CXXFLAGS)"
