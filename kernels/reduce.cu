/**
 * @author Jedges Gyasi\
 * @brief So this reduction kernel will be a two pass reduction kernel on the gpu
 */

#include <iostream>

#define N  1<<22
using namespace std;

__global__ void reduce(float* inputd, float* odata){
    __shared__ int localCache[256];

    int tid = threadIdx.x;
    int Idx = threadIdx.x + blockDim.x * blockIdx.x;

    localCache[tid] = inputd[Idx];
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
    if (tid == 0) odata[blockIdx.x] = localCache[0];
    
}

int main(){
    int threadPerBlock = 256;
    int numBlock = min(32, (N+threadPerBlock-1)/threadPerBlock);
    float a[N];
    float *dev_a, *part_dev_c;

    float s = N*sizeof(float);

    cudaMalloc((void**)&dev_a, s );
    cudaMalloc((void**)&part_dev_c, numBlock*sizeof(float));

    cudaMemcpy(dev_a, a, s, cudaMemcpyHostToDevice);

    reduce <<<numBlock, threadPerBlock>>>(dev_a, part_dev_c);

    float* inp = part_dev_c;
    float* out ;
    int curBlock = numBlock;

    while (curBlock > 1){ // we ensure we only return when we have one block answer remaining which is final answer
        int nextBlock = (curBlock + threadPerBlock-1)/threadPerBlock;
        cudaMalloc(&out, nextBlock*sizeof(float));

        reduce<<<nextBlock, threadPerBlock>>>(inp, out);

        cudaFree(inp);

        inp = out;
        curBlock = nextBlock;
    }

    // single point ans
    float res;

    cudaMemcpy(&res, out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(out);


    // // no need to run final sum on cpu
    // float* part_sum = new float[numBlock];
    // cudaMemcpy(part_sum, part_dev_c, numBlock*sizeof(float), cudaMemcpyDeviceToHost);

    // // finish claculation on cpu which is more effective that on gpu
    // float c = 0.0f;
    // for (int i; i<numBlock; i++){
    //     c += part_sum[i];
    // }

    // delete[] part_sum;
    // cudaFree(dev_a);
    // cudaFree(part_dev_c);

    return 0;

}