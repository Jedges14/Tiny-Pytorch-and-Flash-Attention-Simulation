#include <iostream>
using namespace std;

# define N 10

__global__ void vectoradd(float* A, float* B, float*C){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }
}



int main(){
    // initialize arrays
    int a[N], b[N], c[N];

    int size = N*sizeof(int);

    // device pointers from host to device
    float *dev_a, *dev_b, *dev_c;

    // alloc memory on device memory
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    for(int i=0; i<N; i++){
        a[i] = i;
        b[i] = i;
    }

    // copy value from host to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    vectoradd<<<N, 1>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    // for(int i=0; i<N; i++){
    //     printf("%d + %d = %d\n", a[i], b[i], c[i]);
    // }

    // printf( "%p + %p\n\n", dev_a, dev_b);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}




