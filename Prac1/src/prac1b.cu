//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// From CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

// From the SDK
#include "helper_cuda.h"
#include "helper_string.h"

//
// kernel routine
// 

__global__ void my_first_kernel(float *x, const float* v_1, const float* v_2)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = v_1[tid] + v_2[tid];
}


//
// main code
//

int main(int argc, const char **argv){

  // initialise card
  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block
  int nblocks, nthreads, nsize, n;
  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory in host
  float *h_v1 = (float *)malloc(nsize*sizeof(float));
  float *h_v2 = (float *)malloc(nsize*sizeof(float));
  float *vec_result = (float *)malloc(nsize*sizeof(float));

  // Allocate mem in device
  float *d_x = NULL;
  float *d_v1 = NULL;
  float *d_v2 = NULL;
  checkCudaErrors(cudaMalloc((void **)&d_x, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_v1, nsize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&d_v2, nsize*sizeof(float)));

  // Initialize arrays
  for(n=0; n<nsize; n++){
      h_v1[n] = float(n);
      h_v2[n] = n*2.0;
  }

  // copy input data from host to device
  checkCudaErrors(cudaMemcpy(d_v1,h_v1,nsize*sizeof(float),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_v2,h_v2,nsize*sizeof(float),cudaMemcpyHostToDevice));

  // execute kernel
  my_first_kernel<<<nblocks,nthreads>>>(d_x, d_v1, d_v2);
  getLastCudaError("my_first_kernel execution failed\n");

  // wait for the kernel to finish
  checkCudaErrors(cudaDeviceSynchronize());

  // copy back results and print them out
  checkCudaErrors(cudaMemcpy(vec_result,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost));

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,vec_result[n]);

  // free memory 
  free(h_v1);
  free(h_v2);
  free(vec_result);
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_v1));
  checkCudaErrors(cudaFree(d_v2));

  // CUDA exit -- needed to flush printf write buffer
  cudaDeviceReset();

  return 0;
}
