#include <iostream>
#include <string>
#include <cstdlib>

// 2 GB of data
#define BYTES 2147483648
#define MAX_STRIDE 4194304
#define MAX_INDEX (BYTES/MAX_STRIDE)

static void __global__
Set(const int Seed, const int Stride, char *data) {
	// Everyone set some data
	for(int i = threadIdx.x; i < BYTES ; i+= blockDim.x ) {
		data[i] = (char)(i * threadIdx.x + Seed);
	}
}

static void __global__
CacheTest(const int Seed , const int Stride , char *data, int *result ) {
	if(threadIdx.x == 0 ) {
		int local_result = 0;
		for(int i = 0 ; i < MAX_INDEX; ++i) {
			local_result += data[i *(Stride+1)];
		}
		*result = local_result;	// Here to make sure we don't optimize the loop away
	}
}

static void Validate(bool cond, std::string msg) {
	if(!cond) {
		std::cout << msg << std::endl;
		cudaDeviceReset();
		std::exit(EXIT_FAILURE);
	}
}

static void CudaCheck(std::string msg, cudaError err) {
	Validate(err==cudaSuccess, cudaGetErrorString(err) + std::string("\n") + msg);
}

void sort(float *durations, int size) {
	for(int i = 1 ; i < size; ++i) {
		float cand = durations[i];
		int j = i;
		while( j > 0 && cand < durations[j-1]) {
			if(durations[j] < durations[j-1]) {
				durations[j] = durations[j-1];
			}
			--j;
		}
		durations[j] = cand;
	}
}


float Run(const int Seed, const int Stride) {
	
	const int Blocks = 1;
	const int Threads = 1024;
	float time;
	cudaEvent_t start,end;
	
	char *d_data;
	int *d_result;
	
	CudaCheck("Malloc Result", cudaMalloc(&d_result, sizeof(int) ) );
	CudaCheck("Malloc Data", cudaMalloc(&d_data, BYTES ) );
	
	CudaCheck("Memset Result", cudaMemset(d_result, 0, sizeof(int) ) );
	CudaCheck("Memset Data", cudaMemset(d_data, 0, BYTES ) );
	
	Set<<<Blocks,Threads>>>(Seed, Stride, d_data);
	CudaCheck("Set",cudaDeviceSynchronize());
	
	CudaCheck("Create start",cudaEventCreate(&start));
	CudaCheck("Create end",cudaEventCreate(&end));
	
	
	
	CudaCheck("Record start",cudaEventRecord(start,0));
	CacheTest<<<Blocks,Threads>>>(Seed, Stride, d_data, d_result);
	CudaCheck("Record end",cudaEventRecord(end,0));
	CudaCheck("Device Sync",cudaDeviceSynchronize());
	
	CudaCheck("Event sync", cudaEventSynchronize(end) );
	CudaCheck("Get elapsed time",cudaEventElapsedTime(&time,start,end));
	
	CudaCheck("Destroy start",cudaEventDestroy(start));
	CudaCheck("Destroy end",cudaEventDestroy(end));
	
	
	CudaCheck("Free result", cudaFree(d_result));
	CudaCheck("Free data", cudaFree(d_data));
	
	CudaCheck("Reset",cudaDeviceReset());
	
	return time;
}

int main(int argc, char* argv[]) {
	
	const int Runs = 50;
	float durations[Runs];
	
	Validate(argc==2,"Usage: " + std::string(argv[0]) + " stride");
	const int Stride = atoi(argv[1]);
	Validate(Stride <= MAX_STRIDE,"Decrease Stride");
	
	std::cout << "Stride: " << Stride << std::endl;
	
	for(int i = 0 ; i < Runs ; ++i ) {
		durations[i] = Run(i+1, Stride);
	}
	sort(durations,Runs);
	
	float time = 0;
	int count = 0;
	for(int i = 0; i < Runs; ++i) {
		time += durations[i];
		++count;
	}
	time /= count;
	
	std::cout << "Elapsed Time: " << time << "ms" << std::endl;	
	
	return EXIT_SUCCESS;
}