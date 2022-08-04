#include<fstream>
#include<iostream>
/*
This code is for the test of loading data from .bin file
*/
using namespace std;

int main(){
    ifstream fin("../test_data.bin",ios::binary);
    float *data= new float[10*3*2500];
    float single_data;
    int index=0;

    while (fin.read((char*)&single_data,sizeof(float))){
        data[index*3]=single_data;
        fin.read((char*)&single_data,sizeof(float));
        data[index*3+1]=single_data;
        fin.read((char*)&single_data,sizeof(float));
        data[index*3+2]=single_data;
        index++;
    }
    fin.close();
    cout<<index<<' '<<"yeah"<<endl;
    for (int i=0;i<2;i++){
        index=9*3*2500+2500-2+i;
        cout<<data[index]<<' '<<data[index+2500]<<' '<<data[index+2*2500]<<endl;
    }

    delete []data;
    return 0;
}