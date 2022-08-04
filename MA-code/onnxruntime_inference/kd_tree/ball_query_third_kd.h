#ifndef BALL_QUERY_THIRD_KD
#define BALL_QUERY_THIRD_KD

#include "nanoflann.hpp"
#include "KDTreeTableAdaptor.h"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <omp.h>

#include <vector>
#include <random>
#include <algorithm>
#include <iterator>

/*
This file defines the ball query based on k-d tree structure
First find all neighbors within radius, then evenly sample K elements
 */ 
void cpp_knn_radius(const float* points, const size_t npts, const size_t dim, 
			const float* queries, const size_t nqueries,
			const size_t K, const float radius, int* indices);
// Parameters: Array of point cloud, number of points, number of dimensions, 
//			Array of centers, number of centers, 
//			sample number, radius, indices of sampled point

// the same algorithm but for variable batch size
void batch_cpp_knn_radius(const float* points, const size_t bs, const size_t npts, const size_t dim, 
			const float* queries, const size_t nqueries,
			const size_t K, const float radius, int* indices);
// Parameters: Array of point cloud, batch size, number of points, number of dimensions, 
//			Array of centers, number of centers, 
//			sample number, radius, indices of sampled point

// the same algorithm but for variable batch size and multi-threading
void batch_cpp_knn_radius_omp(const float* points, const size_t bs, const size_t npts, const size_t dim, 
			const float* queries, const size_t nqueries,
			const size_t K, const float radius, int* indices);
// Parameters: Array of point cloud, batch size, number of points, number of dimensions, 
//			Array of centers, number of centers, 
//			sample number, radius, indices of sampled point

using namespace nanoflann;

void cpp_knn_radius(const float* points, const size_t npts, const size_t dim, 
			const float* queries, const size_t nqueries,
			const size_t K, const float radius, int* indices){
	// create the kdtree
	typedef KDTreeTableAdaptor< float, float> KDTree;
	KDTree mat_index(npts, dim, points, 10);
	mat_index.index->buildIndex();

	std::vector<std::pair<size_t, float>> out_pair;
	// iterate over the points
	for(size_t i=0; i<nqueries; i++){
		out_pair.clear();
		// find all neighbors within the radius
		size_t num = mat_index.index->radiusSearch(&queries[i*dim], radius*radius, out_pair, SearchParams(10,0.0,false));//注意已经手动在nanoflann中把排序关掉了
		if (num > 0) {
			// sample from the found points, cyclically repeat with a stride
			size_t stride = std::max(int(num/K), 1);
			for (size_t j=0; j<K; ++j){
				indices[i*K + j] = int(out_pair[(j*stride)%num].first);
			}		
		} else {
			std::cout<<"no point valid"<<std::endl;
			// nothing found, return negative index
			for (size_t j=0; j<K; ++j){
				indices[i*K + j] = -1;
			}
		}
	}	
}


void batch_cpp_knn_radius(const float* points, const size_t bs, const size_t npts, const size_t dim, 
			const float* queries, const size_t nqueries,
			const size_t K, const float radius, int* indices){
	for (size_t i=0; i<bs; ++i){
		// move pointers and call the neighbor finding function
		const float* p = points + i*npts*dim;
		const float* q = queries + i*nqueries*dim;
		int* id = indices + i*nqueries*K;

		cpp_knn_radius(p, npts, dim, q, nqueries, K, radius, id);
	}
}


void batch_cpp_knn_radius_omp(const float* points, const size_t bs, const size_t npts, const size_t dim, 
			const float* queries, const size_t nqueries,
			const size_t K, const float radius, int* indices){
	
	# pragma omp parallel for
	for (size_t i=0; i<bs; ++i){
		// move pointers and call the neighbor finding function
		const float* p = points + i*npts*dim;
		const float* q = queries + i*nqueries*dim;
		int* id = indices + i*nqueries*K;

		cpp_knn_radius(p, npts, dim, q, nqueries, K, radius, id);
	}
}

#endif