#include <iostream>
using namespace std;
#define N 10


__global__ void ma(float *a, float*b, float *c, int width){

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < width && col < width){
        int idx = row * width + col;
        c[idx] = a[idx] + b[idx];
    }
}


int main(){
    float a[N][N], b[N][N], c[N][N];

    float *dev_a, *dev_b, *dev_c;

    size_t size = N*N*sizeof(float);

    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);


    dim3 block(16, 16);
    dim3 grid((N+block.x-1)/block.x, (N+block.y-1)/block.y);

    ma<<<grid, block>>>(dev_a, dev_b, dev_c, N);

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);



    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}