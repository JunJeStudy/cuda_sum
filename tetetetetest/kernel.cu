

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


//#define array_size 100000000
#define array_size 101

//987459712


cudaError_t addWithCuda(int *total);

__shared__ int temp[array_size];

__global__ void addKernel(int *tid_c, int *tid_total)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	tid_c[tid] = tid;
	if (tid <= array_size)
	{
		
			temp[threadIdx.x] = tid;

			if (threadIdx.x==0)
			{
				for(int i=0;i<=blockDim.x;i++)
				{
					//__syncthreads();
					atomicAdd(tid_total, temp[i]);
					//__syncthreads();
					//printf("i = %d \n", *tid_total);
				}
			}

	}

}
int main() 
{

	int	total=0;
	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(&total);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
/*
	for (int i = 0; i < array_size; i++){
		printf("%d - %d\n", i, c[i]);
	}*/
	printf("%d", total);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *total)
{
	int *a;
	int *cuda_total;
	//int *dev_a = 0;
	//int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaMalloc(&a, sizeof(int)*array_size);
	cudaStatus = cudaMalloc(&cuda_total, sizeof(int)); 
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	dim3 dimBlock(32);
	dim3 dimGrid((array_size + dimBlock.x - 1) / dimBlock.x);

	addKernel << <dimGrid, dimBlock >> >(a,cuda_total);

		// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	printf("tetst %d\n", *total);
	cudaStatus = cudaMemcpy(total, cuda_total, sizeof(int), cudaMemcpyDeviceToHost);
	printf("tetst %d\n", *total);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	

Error:
	cudaFree(a);
	cudaFree(cuda_total);

	return cudaStatus;
}