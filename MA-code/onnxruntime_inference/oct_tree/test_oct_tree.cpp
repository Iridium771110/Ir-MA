/*
This file is a test file for octree structure and its fundemental functions
*/
#include "oct_tree.h"
#include <iostream>
#include <vector>

int main(){

    float reso,x0,x1,y0,y1,z0,z1;
    OCT_TREE::Point* points= new Point[10];
    for (int i=0;i<10;i++){
        points[i].x= float(i);
        points[i].y=points[i].x;
        points[i].z=points[i].y;
    }// initialize the test points

    x0=0.0f; y0=0.0f; z0=0.0f;
    x1=9.0f; y1=9.0f; z1=9.0f;
    reso=0.9; // configuration of octree
    OCT_TREE::Oct_Tree oct_tree(reso,x0,x1,y0,y1,z0,z1); //initialize the octree

    int point_labels[10];
    for (int i=0;i<10;i++){
        point_labels[i]=oct_tree.Creat_Tree(points[i]);
        std::cout<<point_labels[i]<<' '<<points[i].x<<' '<<points[i].y<<' '<<points[i].z<<std::endl;
    }// create the octree using the test points and record the corresponding block label for test points

    int valid_leaf_num=oct_tree.valid_node_num;
    OCT_TREE::Point* centers=new Point[valid_leaf_num];
    OCT_TREE::Oct_Node** valid_nodes=new Oct_Node*[valid_leaf_num];
    OCT_TREE::Oct_Node* cur_node;

    oct_tree.Get_all_valid(valid_nodes,valid_leaf_num);// get all valid leaf nodes
    for (int i=0;i<valid_leaf_num;i++){
        cur_node=valid_nodes[i];
        centers[i].x=(cur_node->max_range.x+cur_node->min_range.x)/2.0f;
        centers[i].y=(cur_node->max_range.y+cur_node->min_range.y)/2.0f;
        centers[i].z=(cur_node->max_range.z+cur_node->min_range.z)/2.0f;
        std::cout<<centers[i].x<<' '<<centers[i].y<<' '<<centers[i].z<<std::endl;
    }// check the center of valid leaf nodes
    cur_node=nullptr;

    OCT_TREE::Point a;
    a.x=1.1;a.y=1.1;a.z=1.1;
    std::cout<<oct_tree.Get_position_label(a)<<std::endl;// check if the point can be located in the correct leaf node

    delete []points;
    delete []centers;
    delete []valid_nodes;

    return 0;
}