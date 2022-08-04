#include "ball_query.h"
#include "group_points.h"
#include "interpolate.h"
#include "sampling.h"

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("gather_points", &gather_points);
//   m.def("gather_points_grad", &gather_points_grad);
//   m.def("furthest_point_sampling", &furthest_point_sampling);

//   m.def("three_nn", &three_nn);
//   m.def("three_interpolate", &three_interpolate);
//   m.def("three_interpolate_grad", &three_interpolate_grad);

//   m.def("ball_query", &ball_query);

//   m.def("group_points", &group_points);
//   m.def("group_points_grad", &group_points_grad);
// }

/*
This file is modified from the original project.
The function below registers the custom operators with TorchScript, which let the convertor recognize these operators.
This is one of feasible ways to register the custom operators with TorchScript
*/
TORCH_LIBRARY(pnt2_ops, m){
    m.def("sampling", furthest_point_sampling);
    m.def("gather_points", gather_points);
    m.def("ball_query", ball_query);
    m.def("grouping", group_points);
}