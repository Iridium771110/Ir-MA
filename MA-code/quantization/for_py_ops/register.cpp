#include "register.h"
#include "common.h"
#include "ball_query.h"
#include "gather_points.h"
#include "grouping.h"
#include "sampling.h"
const char *CustomOpsDomain="onnx_pnt2_ops";
SamplingCustomOp onnx_sampling;
GatherPointsCustomOp onnx_gather_points;
GroupingCustomOp onnx_grouping;
BallQueryCustomOp onnx_ball_query;

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api){
    // registration of custom operators with ONNX Runtime
    OrtCustomOpDomain *domain=nullptr;
    
    const OrtApi *ortApi=api->GetApi(ORT_API_VERSION);

    if (auto status = ortApi->CreateCustomOpDomain(CustomOpsDomain, &domain)) {
        return status;
    }
    if (auto status = ortApi->CustomOpDomain_Add(domain, &onnx_sampling)) {
        return status;
    }
    if (auto status = ortApi->CustomOpDomain_Add(domain, &onnx_gather_points)) {
        return status;
    } 
    if (auto status = ortApi->CustomOpDomain_Add(domain, &onnx_ball_query)) {
        return status;
    }
    if (auto status = ortApi->CustomOpDomain_Add(domain, &onnx_grouping)) {
        return status;
    } 

    return ortApi->AddCustomOpDomain(options,domain);
};