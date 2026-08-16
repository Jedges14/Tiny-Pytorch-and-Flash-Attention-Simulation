/**
 * @author Jedges Gyasi
 * Simple histogram kernel using shared local histograms and perform calculations on global data while storing in local shared histograms
 */

#include <iostream>
#include <cuda.h>

using namespace std;
#define BINS 256
#define N 1<<10

__global__ void histogram(long* idata, int* histogram, long n){
    __shared__ int localhist[BINS];
    // Initialize the temp histogram
    localhist[threadIdx.x] = 0;
    __syncthreads();

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while(tid < n){
        atomicAdd(&localhist[idata[tid]], 1);
        tid += offset;
    }
    __syncthreads();

    atomicAdd(&histogram[threadIdx.x], localhist[threadIdx.x]);
}


int main(){
    unsigned int hist[256];
    long a[N];

    long *dev_a;
    int *dev_hist;

    //use the device gpu grid to for the calculations
    cudaDeviceProp p;
    cudaGetDeviceProperties (&p, 0);
    int numblock = p.multiProcessorCount;
    int s = N*sizeof(long);

    cudaMalloc((void**)&dev_a, s);
    cudaMemcpy(dev_a, a, s, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_hist, 256*sizeof(long));
    cudaMemset(&dev_hist, 0, 256*sizeof(long));

    histogram<<<numblock,BINS>>>(dev_a, dev_hist, N);

    cudaMemcpy(hist, dev_hist, 256*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_hist);

    return 0;
}