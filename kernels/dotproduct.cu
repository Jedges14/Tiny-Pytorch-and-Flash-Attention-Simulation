/**
 * @author Jedges Gyasi\
 * @brief So this reduction kernel will be a two pass reduction kernel on the gpu
 */

#include <iostream>
#include <cuda.h>
#define threadsperblock 256
#define N 1<<10
__global__ void dot(float* a, float* b, int* out, int n){
    __shared__ int cache[256];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int Idx = threadIdx.x;
    
    cache[Idx] = 0;
    __syncthreads();

    while(tid < n){
        cache[Idx] += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    __syncthreads();

    //perform redux on the final array
    int i = blockDim.x/2;
    while (i != 0){
        if (Idx < i){
            cache[Idx] += cache[Idx + i];
        }
        __syncthreads();
        i/=2;
    }

    if (Idx==0) atomicAdd(out, cache[0]);
}


int main(){
    int numblocks = min(32, (N+threadsperblock-1)/threadsperblock);

    float a[N], b[N];
    float *dev_a, *dev_b;
    int *final_val;
    long s = N * sizeof(int);

    cudaMalloc((void**)&dev_a, s);
    cudaMalloc((void**)&dev_b, s);

    cudaMemset(&final_val, 0, sizeof(int));

    cudaMemcpy(dev_a, a, s, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, s, cudaMemcpyHostToDevice);

    dot<<<numblocks, threadsperblock>>>(dev_a, dev_b, final_val, N);

    int res;
    cudaMemcpy(&res, final_val, s, cudaMemcpyDeviceToHost);

    cudaFree(final_val);

    return 0;
}