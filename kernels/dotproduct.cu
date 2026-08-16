/**
 * @author Jedges Gyasi\
 * @brief So this reduction kernel will be a two pass reduction kernel on the gpu
 */

#include <iostream>
#include <cuda.h>

__global__ void dot(int* a, int* b, int* out, int n){
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