#ifndef REGISTER
#define REGISTER
#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C"{
#endif

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api); // registration of custom operators with ONNX Runtime

#ifdef __cplusplus
}
#endif

#endif