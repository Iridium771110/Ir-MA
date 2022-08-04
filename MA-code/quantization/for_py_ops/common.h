#ifndef COMMON
#define COMMON

#include <iostream>
#include "onnxruntime_cxx_api.h"
#include <omp.h>

struct INPUT{
    char* name;
    float* value;
    int num;
};

struct OrtTensorDimensions : std::vector<int64_t> {
    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue *value){
        OrtTensorTypeAndShapeInfo *info=ort.GetTensorTypeAndShape(value);
        std::vector<int64_t>::operator=(ort.GetTensorShape(info));
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
};

#endif