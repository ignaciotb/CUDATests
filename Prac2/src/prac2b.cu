
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <time.h>
#include <bits/time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

__constant__ float a, b, c;
__constant__ int N;

__global__ void polyAverage(const float* d_z, float* d_v){

    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    d_v[tid] = 0.0;
    int th_cnt;
    for (int i=0; i<100; i++) {
        th_cnt = tid*N + i;
        d_v[tid] += a * (d_z[th_cnt]*d_z[th_cnt]) + b * d_z[th_cnt] + c;
    }

    d_v[tid] = d_v[tid] / N;
}


int main(int argc, const char **argv){

    // initialise card
    findCudaDevice(argc, argv);
    int NPATH=6400, h_N=100;

    // Alloc mem in host
    float* h_out = (float *)malloc(sizeof(float)*NPATH);
    float* h_z = (float *)malloc(sizeof(float)*h_N*NPATH);
    float* h_d = (float *)malloc(sizeof(float)*h_N*NPATH);

    // Alloc mem in device
    float* d_z = NULL;
    float* d_v = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_v, sizeof(float)*NPATH));
    checkCudaErrors(cudaMalloc((void **)&d_z, sizeof(float)*h_N*NPATH));

    // Store const vars in device mem
    float h_a, h_b, h_c;
    h_a = 10;
    h_b = 100;
    h_c = 1000;
    checkCudaErrors(cudaMemcpyToSymbol(a, &h_a, sizeof(h_a)));
    checkCudaErrors(cudaMemcpyToSymbol(b, &h_b, sizeof(h_b)));
    checkCudaErrors(cudaMemcpyToSymbol(c, &h_c, sizeof(h_c)));
    checkCudaErrors(cudaMemcpyToSymbol(N, &h_N, sizeof(h_N)));

    // initialise CUDA timing
    float milliGPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // random number generation
    curandGenerator_t gen;
    checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
    checkCudaErrors( curandGenerateNormal(gen, d_z, h_N*NPATH, 0.0f, 1.0f) );

    // Launch kernel
    cudaEventRecord(start);

    polyAverage<<<NPATH/32, 32>>>(d_z, d_v);
    getLastCudaError("Exec of average of poly kernel failed \n");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliGPU, start, stop);
    printf("Average of polynomial GPU execution time (ms): %f \n", milliGPU);

    // Copy result from device
    checkCudaErrors(cudaMemcpy(h_out, d_v, sizeof(float)*NPATH, cudaMemcpyDeviceToHost));

    // dealloc mem in device
    checkCudaErrors(cudaFree(d_z));
    checkCudaErrors(cudaFree(d_v));

    // compute average
    double sum1, sum2;
    sum1 = 0.0;
    sum2 = 0.0;
    for (int i=0; i<NPATH; i++) {
      sum1 += h_out[i];
      sum2 += h_out[i]*h_out[i];
    }

    printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
       sum1/NPATH, sqrt((sum2/NPATH - (sum1/NPATH)*(sum1/NPATH))/NPATH) );

    checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
    checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
    checkCudaErrors(curandGenerateNormal(gen, h_z, h_N*NPATH, 0.0f, 1.0f));

    // Compute average of poly in CPU
//    clock_t tStart = clock();
//    long double sum3, sum4;
//    for(int i=0; i<NPATH*h_N; i++){
//        h_d[i] = h_a * (h_z[i]*h_z[i]) + h_b * h_z[i] + h_c;
//    }

//    printf("Average of polynomial CPU execution time (ms): %.4fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

//    for(int i=0; i<NPATH*h_N; i++){
//        sum3 += h_d[i];
//        sum4 += std::pow(h_d[i], 2);
//    }

//    std::cout << "Average value : " << sum3/double(NPATH*h_N) << std::endl;
//    printf("\nAverage value and standard deviation of error  = %13.8f %13.8f\n\n",
//    sum3/(NPATH*h_N), sqrt((sum4/(NPATH*h_N) - (sum3/(NPATH*h_N))*(sum3/(NPATH*h_N)))/(NPATH*h_N)) );

    // clean up curand
    checkCudaErrors( curandDestroyGenerator(gen));

    // dealloc mem in host
    free(h_out);
    free(h_z);
    free(h_d);

    // CUDA exit
    cudaDeviceReset();

}
