//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <iostream>
#include <vector>

// From CUDA headers
#include <cuda.h>   // Defines __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// From the SDK
#include "helper_cuda.h"
#include "helper_string.h"

//
// kernel routine
// 

__global__ void my_first_kernel(float *x, unsigned int y)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  printf("%d \n", tid);
  x[tid] = (float) threadIdx.x + y;
}

//
// main code
//

int main(int argc, const char **argv)
{
  float *x;
  int nblocks, nthreads, nsize, n;

  // Initialise card
  cudaDeviceProp deviceProp;
  int devID = findCudaDevice(argc, argv);
  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

  // set number of blocks, and threads per block
  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array
  checkCudaErrors(cudaMallocManaged(&x, nsize*sizeof(float)));

  // execute kernel
  my_first_kernel<<<nblocks,nthreads>>>(x, 1);
  getLastCudaError("my_first_kernel execution failed\n");

  // synchronize to wait for kernel to finish, and data copied back
  checkCudaErrors(cudaDeviceSynchronize());
  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,x[n]);

  // free memory 
  checkCudaErrors(cudaFree(x));

  // CUDA exit -- needed to flush printf write buffer
  cudaDeviceReset();

  return 0;
}
