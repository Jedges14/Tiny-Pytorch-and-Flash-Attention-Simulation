#include <iostream>
using namespace std;
#define N 10


__global__ void SAXPY(float a, float* x, float* y, int n){
    int tid = threadIdx.x  + blockIdx.x * blockDim.x;

    while (tid < n){
        y[tid] += a * x[tid];
        tid += blockDim.x * gridDim.x;
    }
}


int main(){
    float a[N], b[N];
    float *dev_a, *dev_b;

    int s = N * sizeof(float);

    cudaMalloc((void**)&dev_a, s);
    cudaMalloc((void**)&dev_b, s);

    cudaMemcpy(dev_a, a, s, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, s, cudaMemcpyHostToDevice);

    SAXPY<<<4, N>>>(2.0, dev_a, dev_b, N);

    cudaMemcpy(b, dev_b, s, cudaMemcpyDeviceToHost);

    return 0;



}