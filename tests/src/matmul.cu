
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "helper_cuda.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    float * elements;
} Matrix;

__global__ void matMulNaiveKernel(Matrix A, Matrix B, Matrix C){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float Cvalue;
    for(int i = 0; i< A.width; i++){
        Cvalue += A.elements[row * A.width + i] * B.elements[i * B.width + col];
    }
    C.elements[row * C.width + col] = Cvalue;
}



void matMul(const Matrix A, const Matrix B, Matrix C){

    // Allocate device mem
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    checkCudaErrors(cudaMalloc((void **)&d_A.elements, sizeof(float)*A.width*A.height));

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    checkCudaErrors(cudaMalloc((void **)&d_B.elements, sizeof(float)*B.width*B.height));

    Matrix d_C;
    d_C.width = C.width;
    d_C.height = C.height;
    checkCudaErrors(cudaMalloc((void **)&d_C.elements, sizeof(float)*C.width*C.height));

    // Transfer mem to device
    checkCudaErrors(cudaMemcpy(d_A.elements, A.elements, sizeof(float)*A.width*A.height, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B.elements, B.elements, sizeof(float)*B.width*B.height, cudaMemcpyHostToDevice));

    // Initialise CUDA timing
    float milliGPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width/dimBlock.x, A.height/dimBlock.y);

    cudaEventRecord(start);

    matMulNaiveKernel<<<dimGrid, dimBlock >>>(d_A, d_B, d_C);
    getLastCudaError("Exec of matMulKernel failed \n");

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliGPU, start, stop);
    printf("GPU execution time (ms): %f \n", milliGPU);

    // Copy mem device to host
    checkCudaErrors(cudaMemcpy(C.elements, d_C.elements, sizeof(float)*C.width*C.height, cudaMemcpyDeviceToHost));

    // Free device mem
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

}

int main(){

    // Allocate A in host
    Matrix A;
    A.width = 1000;
    A.height = 2000;
    A.elements = (float *)malloc(sizeof(float)*A.width*A.height);

    for(int i=0; i<A.height*A.width; i++){
        A.elements[i] = i;
    }

    // Allocate B in host
    Matrix B;
    B.width = 2000;
    B.height = 500;
    B.elements = (float *)malloc(sizeof(float)*B.width*B.height);

    for(int i=0; i<B.height*B.width; i++){
        B.elements[i] = i;
    }

    // Allocate C in host
    Matrix C;
    C.height = A.height;
    C.width = B.width;
    C.elements = (float *)malloc(sizeof(float)*C.width*C.height);

    matMul(A, B, C);

    // Free mem in host
    free(A.elements);
    free(B.elements);

    return 0;
}












