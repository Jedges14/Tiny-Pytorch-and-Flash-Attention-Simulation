/**
 * @author Jedges Gyasi\
 * @brief So this reduction kernel will be a two pass reduction kernel on the gpu
 */

#include <iostream>

#define N  1<<22
using namespace std;

__global__ void reduce(float* inputd, float* odata, int n){
    __shared__ int localCache[256];

    int tid = threadIdx.x;
    int Idx = threadIdx.x + blockDim.x * blockIdx.x;

    localCache[tid] = Idx<n ? inputd[Idx] : 0;
    __syncthreads(); // all threads can now see data and begin calculation

    int i = blockDim.x/2;
    while (i != 0){
        if (tid < i){
            localCache[tid] += localCache[tid + i];
        }
        __syncthreads();
        i/=2;
    }
    // map block that did the caculation on global memory
    if (tid == 0) atomicAdd(odata, localCache[0]);
}



int main(){
    int threadPerBlock = 256;
    int numBlock = min(32, (N+threadPerBlock-1)/threadPerBlock);
    float a[N];
    float *dev_a, *dev_out;

    float s = N*sizeof(float);

    cudaMalloc((void**)&dev_a, s );
    cudaMalloc((void**)&dev_out,sizeof(float));
    cudaMemset(dev_out, 0, sizeof(float));

    cudaMemcpy(dev_a, a, s, cudaMemcpyHostToDevice);

    reduce <<<numBlock, threadPerBlock>>>(dev_a, dev_out, N);

    float res;

    cudaMemcpy(&res, dev_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_out);
    return 0;

}