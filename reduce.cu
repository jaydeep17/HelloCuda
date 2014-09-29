#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

#include "timer.h"

/**
 * A parallel reduce where array is divided into two parts and the right hand side
 * part is added to the left hand side half, and so on.
 * Eg: [ 1 2 3 4 5 6 7 8 ]
 * Step 1 => [ 6 8 10 12 5 6 7 8 ]
 * Step 2 => [ 16 20 10 12 5 6 7 8 ]
 * Step 3 => [ 36 20 10 12 5 6 7 8 ]
 * Thus answer is 36
 */
__global__ void reduceContiguous(int *d_in, int *d_out) {

	extern __shared__ int sdata[];

	sdata[threadIdx.x] = d_in[threadIdx.x];
	__syncthreads();

	int myId = threadIdx.x + (blockIdx.x * blockDim.x);
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s)
			sdata[myId] += sdata[myId + s];
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		*d_out = sdata[(blockIdx.x * blockDim.x)];
	}
}


/**
 * A parallel reduce where each array element is added to it's adjacent result
 * Eg: [ 1 2 3 4 5 6 7 8 ]
 * Step 1 => [ 3 2 7 4 11 6 15 8 ]
 * Step 2 => [ 10 2 7 4 26 6 15 8 ]
 * Step 3 => [ 36 2 7 4 26 6 15 8 ]
 * Thus answer is 36
 */
__global__ void reduceInterleaving(int *d_in, int *d_out) {

	extern __shared__ int sdata[];

	int tid = threadIdx.x;
	int myId = threadIdx.x + (blockIdx.x * blockDim.x);
	sdata[threadIdx.x] = d_in[threadIdx.x];
	__syncthreads();

	for (int i = 1, v = 2; i < blockDim.x; i *= 2, v *= 2) {
		if ( (tid % (v)) == 0 )
			sdata[tid] += sdata[tid + i];
		__syncthreads();
	}

	if (myId == 0) {
		*d_out = sdata[0];
	}
}


/**
 * Driver function to allocate/initialize all the memory and free it
 * after kernel exits
 */
void _callReduce(bool contiguous) {
	// NOTE : here N can't be greater than 1024
	const int N = 1024;
	int h_in[N], h_out;
	for (int i = 0; i < N; ++i) {
		h_in[i] = i;
	}

	int *d_in, *d_out;
	cudaMalloc((void**) &d_in, sizeof(h_in));
	cudaMalloc((void**) &d_out, sizeof(h_out));

	cudaMemcpy(d_in, h_in, sizeof(int) * N, cudaMemcpyHostToDevice);

	GpuTimer timr;
	timr.Start();
	if (contiguous)
		reduceContiguous<<<1, N, sizeof(h_in)>>>(d_in, d_out);
		else
		reduceInterleaving<<<1, N, sizeof(h_in)>>>(d_in, d_out);

	cudaMemcpy(&h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
	timr.Stop();
	float elap = timr.Elapsed();

	printf("sum = %d in time %f\n", h_out, elap);

	cudaFree(d_in);
	cudaFree(d_out);
}


void callReduceContiguous() {
	_callReduce(true);
}

void callReduceInterleaving() {
	_callReduce(false);
}
