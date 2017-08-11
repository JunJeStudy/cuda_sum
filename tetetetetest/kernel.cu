

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


//#define array_size 100000000
#define array_size 100

//987459712


cudaError_t addWithCuda(int *c, unsigned int size);

__device__ int temp[array_size];

__global__ void addKernel(int *c)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	temp[tid] += tid + 1;
	int i_sum = tid;

	if (i_sum % 2 == 0){ i_sum /= 2; }
	else { i_sum /= 2 + 1; }

	for (int i = i_sum; i > 0;){
		temp[tid] += temp[i];
		i_sum /= 2;
		__syncthreads();
	}

	c[tid] = temp[tid];
}
int main()
{
	int a[array_size];
	memset(a, 0, array_size * sizeof(int));
	int c[array_size] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, array_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	for (int i = 0; i < array_size; i++){
		printf("%d - %d\n", i, c[i]);
	}

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
cudaError_t addWithCuda(int *c, unsigned int size)
{
	int *dev_a = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <10, 10 >> >(dev_c);

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
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);

	return cudaStatus;
}