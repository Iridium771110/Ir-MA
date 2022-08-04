#ifndef COMMON
#define COMMON
/*
This file includes the common part of other head files
*/
#include <iostream>
#include "onnxruntime_cxx_api.h"

struct INPUT{
    char* name;
    float* value;
    int num;
}; // seems not used in the end

struct OrtTensorDimensions : std::vector<int64_t> {
    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue *value){
        OrtTensorTypeAndShapeInfo *info=ort.GetTensorTypeAndShape(value);
        std::vector<int64_t>::operator=(ort.GetTensorShape(info));
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
}; // to extract the dimensional information of a tensor

#endif