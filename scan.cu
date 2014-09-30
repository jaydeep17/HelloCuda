#include <stdio.h>
#include <stdlib.h>

#include "timer.h"


__global__ void HellisScan(int *d_in, int *d_out) {
	int t = threadIdx.x;

	extern __shared__ int sdata[];

	sdata[t] = d_in[t];

	for (int i = 1; i < blockDim.x; i *= 2) {
		__syncthreads();
		if (t >= i) {
			sdata[t] += sdata[t - i];
		}
	}

	d_out[t] = sdata[t];
}


void callHellisScan() {
	const int N = 1024;
	int h_in[N], h_out[N];
	for (int i = 0; i < N; ++i) {
		h_in[i] = 1;
	}

	int *d_in, *d_out;
	cudaMalloc((void**) &d_in, sizeof(h_in));
	cudaMalloc((void**) &d_out, sizeof(h_out));

	cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice);

	GpuTimer timr;
	timr.Start();

	HellisScan<<<1, N, sizeof(h_in)>>>(d_in, d_out);

	cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
	timr.Stop();

	if (N > 50) {
		printf("The resultant array is large "
				"so showing the first 50 entries\n");
	}
	for (int i = 0; i < 50; ++i) {
		printf("%d ", h_out[i]);
	}
	printf("\n");
	printf("finished in %f\n", timr.Elapsed());

	cudaFree(d_in);
	cudaFree(d_out);
}
