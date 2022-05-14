#ifndef COMMON
#define COMMON

#include <iostream>
#include "onnxruntime_cxx_api.h"

struct INPUT{
    char* name;
    float* value;
    int num;
};

struct OrtTensorDimensions : std::vector<int64_t> {
    OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue *value){
        OrtTensorTypeAndShapeInfo *info=ort.GetTensorTypeAndShape(value);
        std::vector<int64_t>::operator=/*该操作相当于重置该vector,调用赋值构造函数*/(ort.GetTensorShape(info));
        ort.ReleaseTensorTypeAndShapeInfo(info);
    }
};

#endif