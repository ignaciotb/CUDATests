
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "helper_cuda.h"
#include "helper_string.h"

#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#if __CUDACC_VER_MAJOR__ >= 9
#undef __CUDACC_VER__
#define __CUDACC_VER__ \
  ((__CUDACC_VER_MAJOR__ * 10000) + (__CUDACC_VER_MINOR__ * 100))
#endif
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    float * elements;
    int stride;
} Matrix;


// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, float value){
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is located col sub-matrices to the right and row sub-matrices down from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col){
    Matrix Asub; Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

__global__ void matMulSharedMemKernel(Matrix A, Matrix B, Matrix C){

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    float Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for(int m = 0; m < (A.width / BLOCK_SIZE); m++){

        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded before starting the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();

    }

    // Write Csub to device memory each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

__global__ void matMulNaiveKernel(Matrix A, Matrix B, Matrix C){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float Cvalue;
    for(int i = 0; i< A.width; i++){
        Cvalue += A.elements[row * A.width + i] * B.elements[i * B.width + col];
    }
    C.elements[row * C.width + col] = Cvalue;
}

void matMulNaive(const Matrix A, const Matrix B, Matrix C){

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

    // Invoke naive kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width/dimBlock.x, A.height/dimBlock.y);

    matMulNaiveKernel<<<dimGrid, dimBlock >>>(d_A, d_B, d_C);
    getLastCudaError("Exec of naive matMulKernel failed \n");

    // Copy mem device to host
    checkCudaErrors(cudaMemcpy(C.elements, d_C.elements, sizeof(float)*C.width*C.height, cudaMemcpyDeviceToHost));

    // Free device mem
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

void matMulShared(const Matrix A, const Matrix B, Matrix C){


    // Allocate device mem
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    checkCudaErrors(cudaMalloc((void **)&d_A.elements, sizeof(float)*A.width*A.height));

    Matrix d_B;
    d_B.width = d_B.stride =  B.width;
    d_B.height = B.height;
    checkCudaErrors(cudaMalloc((void **)&d_B.elements, sizeof(float)*B.width*B.height));

    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    checkCudaErrors(cudaMalloc((void **)&d_C.elements, sizeof(float)*C.width*C.height));

    // Transfer mem to device
    checkCudaErrors(cudaMemcpy(d_A.elements, A.elements, sizeof(float)*A.width*A.height, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B.elements, B.elements, sizeof(float)*B.width*B.height, cudaMemcpyHostToDevice));

    // Invoke naive kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width/dimBlock.x, A.height/dimBlock.y);

    matMulSharedMemKernel<<<dimGrid, dimBlock >>>(d_A, d_B, d_C);
    getLastCudaError("Exec of naive matMulKernel failed \n");

    // Copy mem device to host
    checkCudaErrors(cudaMemcpy(C.elements, d_C.elements, sizeof(float)*C.width*C.height, cudaMemcpyDeviceToHost));

    // Free device mem
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

void eigenToArray(float* data, const Eigen::MatrixXd& mat_A)
{
    for (int i = 0; i < mat_A.rows(); ++i){
        for (int j = 0; j < mat_A.cols(); ++j){
            data[i] = mat_A(i,j);
        }
    }
}

int main(){

    // Allocate A in host
    Matrix A;
    A.width = 100;
    A.height = 200;
    A.elements = (float *)malloc(sizeof(float)*A.width*A.height);

    // Allocate B in host
    Matrix B;
    B.width = 200;
    B.height = 500;
    B.elements = (float *)malloc(sizeof(float)*B.width*B.height);

    // Allocate C in host
    Matrix C;
    C.height = A.height;
    C.width = B.width;
    C.elements = (float *)malloc(sizeof(float)*C.width*C.height);

    // Initialise CUDA timing
    float milliGPUShared;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch first version
    cudaEventRecord(start);

    for(int i=0; i<10; i++){
        for(int i=0; i<A.height*A.width; i++){
            A.elements[i] = i;
        }

        for(int i=0; i<B.height*B.width; i++){
            B.elements[i] = i;
        }

        for(int i=0; i<B.height*B.width; i++){
            B.elements[i] = i;
        }

        matMulNaive(A, B, C);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliGPUShared, start, stop);
    printf("GPU execution 1 time (ms): %f \n", milliGPUShared);

    // Launch second version
    cudaEventRecord(start);

    for(int i=0; i<10; i++){
        for(int i=0; i<A.height*A.width; i++){
            A.elements[i] = i;
        }

        for(int i=0; i<B.height*B.width; i++){
            B.elements[i] = i;
        }

        for(int i=0; i<B.height*B.width; i++){
            B.elements[i] = i;
        }

        matMulShared(A, B, C);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliGPUShared, start, stop);
    printf("GPU execution 2 time (ms): %f \n", milliGPUShared);

    // Launch CPU version
    Eigen::MatrixXd AEigen = Eigen::MatrixXd::Random(A.height, A.width);
    Eigen::MatrixXd BEigen = Eigen::MatrixXd::Random(B.height, B.width);
    Eigen::MatrixXd CEigen = Eigen::MatrixXd(A.height, B.width);
    cudaEventRecord(start);

    for(int i=0; i<10; i++){
         CEigen = AEigen * BEigen;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliGPUShared, start, stop);
    printf("CPU execution time (ms): %f \n", milliGPUShared);

    // Free mem in host
    free(A.elements);
    free(B.elements);
    free(C.elements);

    return 0;
}












