
// NON-CUDA
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <exception>

// CUDA
#include <curand_kernel.h>

// BUFFER SIZE
#define TileSize 1024
#define MAX_THREADS 1024

#if TileSize == 32
#define LogTileSize 5
#elif TileSize == 64
#define LogTileSize 6
#elif TileSize == 128
#define LogTileSize 7
#elif TileSize == 256
#define LogTileSize 8
#elif TileSize == 512
#define LogTileSize 9
#elif TileSize == 1024
#define LogTileSize 10
#else
#define LogTileSize 11
#endif

// CPU POINTERS TO GPU ADDRESSES
static float *device_pointer;
static int *device_climbs;
static int *device_best;

// GPU MEMORY ADDRESSES
static __device__ int *climbs_d;
static __device__ int *length_d;
static __device__ __constant__ int swaps_d;
static __device__ int index_d = 0;

// SHARED SCRATCH MEMORY
__shared__ float shared_x[TileSize];
__shared__ float shared_y[TileSize];
__shared__ int shared_weight[TileSize];


static __device__ int nextIndex() {
	
	if(threadIdx.x==0) {
		shared_weight[0] = atomicAdd(&index_d,1);
	}__syncthreads();
	return shared_weight[0];
}

// Calculates euclidean distance given two indices to the points in data
static __device__ int distv(const float* data, int i, int j) {
	float dx = data[2*i] - data[2*j];
	float dy = data[2*i+1] - data[2*j+1];
	
	dx = dx * dx;
	dy = dy * dy;
	
	return __float2int_rn(sqrtf(dx + dy));
}

// Calculates euclidean distance given two points (x1,y1) and (x2,y2)
static __device__ int dist(float &x1, float &x2, float &y1, float &y2) {
	float dx = x1 - x2;
	float dy = y1 - y2;
	
	dx = dx * dx;
	dy = dy * dy;
	
	return __float2int_rn(sqrtf(dx + dy));
}

// allocate and populate memory
static __device__ bool initMemory(const float *data_d,float* &data, int* &weight,const int &cities) {	
	__shared__ float *t_f;
	__shared__ int *t_i;
		
	if(threadIdx.x==0) {
		t_f = new float[2 * (cities + 1)];
		if(t_f!=NULL) {
			t_i = new int[cities];
			if(t_i==NULL) {
				delete t_f;
				t_f = NULL;
			}
		}
	}__syncthreads();
	
	if(t_f == NULL) return false;
	
	data = t_f;
	weight = t_i;
	
	for(int i = threadIdx.x; i < 2*cities; i += blockDim.x) {
		data[i] = data_d[i];
	}__syncthreads();
	
	return true;
}

// Release memory back to GPU
static __device__ void releaseMemory(float* &data, int* &weight) {
	if(threadIdx.x == 0) {
		delete data;
		delete weight;
	}
}

// Permute our path
static __device__ void permute(float* &data, int* &weight, const int &cities, const int &iter) {
	// Permute
	if(threadIdx.x==0) {
		curandState rndstate;
		curand_init(blockIdx.x + gridDim.x * iter, 0, 0, &rndstate);
		for (int i = 1; i < cities; i++) {
			int j = curand(&rndstate) % (cities - 1) + 1;
			
			float x = data[2*i+0];
			float y = data[2*i+1];
			
			data[2*i+0] = data[2*j+0];
			data[2*i+1] = data[2*j+1];
			
			data[2*j+0] = x;
			data[2*j+1] = y;
		}
		data[2*cities+0] = data[0];
		data[2*cities+1] = data[1];
	}__syncthreads();
	
	// Compute new Distances
	for(int i = threadIdx.x; i < cities; i += blockDim.x) {
		weight[i] = distv(data,i, i+1);
	}__syncthreads();
}

// Load data and weight into shared memory for quick use across multiple threads.
static __device__ void loadShared(const float* data, const int* weight, const int &bound) {
	for(int i = threadIdx.x; i < TileSize; i += blockDim.x) {
		int index = i + bound;
		if(index > 0) {
			shared_x[i] = data[2 * index + 0];
			shared_y[i] = data[2 * index + 1];
			shared_weight[i] = weight[index];
		}
	}__syncthreads();
}

// Perform a single iteration of 2opt
static __device__ void singleTwoOpt(const float* data, const int* weight, const int &cities, int &best, int &best_a, int &best_b) {

	for(int front = 0 ; front < cities - 2; front += blockDim.x) {
	
		float c1x,c2x,c4x;
		float c1y,c2y,c4y;
		
		int d12 = 0;
		
		int city = front + threadIdx.x;
		
		if(city < cities - 2) {
		
			c1x = data[2*city];
			c1y = data[2*city + 1];
			
			c2x = data[2*city + 2];
			c2y = data[2*city + 3];
			
			c4x = data[0];
			c4y = data[1];
			
			d12 = weight[city];
		}
	
		for(int back = cities - 1; back > front + 1; back -= TileSize) {
		
			int bound = back - TileSize + 1;
			
			loadShared(data,weight,bound);
			
			int lower = max(bound,city+2);
			
			for(int j = back; j >= lower; --j) {
				int jm = j - bound;
				
				register float c3x = shared_x[jm];
				register float c3y = shared_y[jm];
				
				int candidate = d12 + shared_weight[jm] - (dist(c1x,c3x,c1y,c3y) + dist(c2x,c4x,c2y,c4y));
				
				if(candidate > best) {
					best = candidate;
					best_a = city;
					best_b = j;
				}
				
				// Reuse
				c4x = c3x;
				c4y = c3y;
				
			}__syncthreads(); // done using shared memory, but maybe not others
		}
	}
}

// Propagate maximum value to shared_weight[0]
static __device__ void maximum(int t_val, const int &cities) {
	int index = threadIdx.x % TileSize;
	shared_weight[index] = t_val;
	__syncthreads();

	for(int i = 1 ; i < (MAX_THREADS / TileSize); ++i) {
		if(t_val > shared_weight[index]) {
			shared_weight[index] = t_val;
		}__syncthreads();
	}
	
	index = min(cities,min(blockDim.x,TileSize));
	
	#pragma unroll
	for(int i = 1 ; i < LogTileSize ; ++i ) {
		index = (index + 1) / 2;
		if(threadIdx.x < index) {
			int tmp = shared_weight[threadIdx.x + index];
			if(t_val < tmp) {
				shared_weight[threadIdx.x] = t_val = tmp;
			}
		}__syncthreads();
	}
}

// Reverse the arrays
static __device__ void reverse(float* &data, int* &weight, int start, int end) {
	while(start<end) {
	
		int   w = weight[start+0];
		float x = data[2*start+0];
		float y = data[2*start+1];
		
		weight[start+0] = weight[end-1];
		data[2*start+0] = data[2*end+0];
		data[2*start+1] = data[2*end+1];
		
		
		weight[end-1] = w;
		data[2*end+0] = x;
		data[2*end+1] = y;
	
		start += blockDim.x;
		end -= blockDim.x;
		
	}__syncthreads();
}

// update connections and weights.  If no update possible returns false, otherwise true.
static __device__ bool update(float* &data, int* &weight,const int &cities,int &best, int &front, int &back) {
	
	// Get maximum best from all threads
	maximum(best,cities);
	
	// If none then jump
	if(shared_weight[0] == 0) {
		return false;
	}
	
	for(int i = 0 ; i < swaps_d; ++i) {
		// Get all the data needed
		if(shared_weight[0]==best) {
			shared_weight[1] = threadIdx.x;
		}__syncthreads();
		
		if(threadIdx.x==shared_weight[1]) {
			best = 0;
			shared_weight[2] = front;
			shared_weight[3] = back;
		}__syncthreads();
		
		if(!(back < shared_weight[2]-1) && !(front > shared_weight[3]+1)) {
			best = 0;
		}
		
		
		reverse(data,weight,shared_weight[2]+threadIdx.x+1,shared_weight[3]-threadIdx.x);
		
		// I can split this across two threads!
		if(threadIdx.x<2) {
			int s = shared_weight[2+threadIdx.x];
			weight[s] = distv(data,s,s+1);
		}__syncthreads();
		
		maximum(best,cities);
		
		// If none then jump
		if(shared_weight[0] == 0) {
		break;
		}
		
	}
	best = 0;
	
	return true;
}

static __device__ void saveResults(int climbs, int best) {
	if(threadIdx.x == 0 ) {
		atomicAdd(climbs_d,climbs);
		atomicMin(length_d,best);
	}
}

static __global__ void __launch_bounds__(MAX_THREADS,2) IterativeTwoOpt(const float* data_d, const int cities,const int restarts, int *climbs, int *len) {
	// save to global
	length_d = len;
	climbs_d = climbs;
	
	float* data;
	int *weight;
	int local_climbs = 0;
	int length = INT_MAX;
	
	// Create memory for block
	if(!initMemory(data_d,data,weight,cities)) {
		if(threadIdx.x==0){
			printf("Error allocating memory for block %d\n",blockIdx.x);
		}
		return;
	}
	
	// Multiple random paths per block
	for(int iter = nextIndex(); iter < restarts; iter = nextIndex()) {	
		int best=0,front=0,back=0;
		
		// Randomize our path
		permute(data,weight,cities,iter);

		// Perform 2-opt until no new changes can be made
		do {
			singleTwoOpt(data,weight,cities,best,front,back);
			++local_climbs;
		}while(update(data,weight,cities,best,front,back));
		
		// Calculate the path
		shared_weight[0] = 0;
		__syncthreads();
		
		best = 0;
		for(int i = threadIdx.x ; i < cities; i+= blockDim.x) {
			best += weight[i];
		}atomicAdd(shared_weight,best);
		__syncthreads();
		
		// Save if better
		if( threadIdx.x==0 && shared_weight[0] < length) {
			length  = shared_weight[0];
		}
	}
	
	// finally done so save our best ones
	saveResults(local_climbs,length);	
	releaseMemory(data,weight);
}





///////////////
///// CPU /////
///////////////

static void check(cudaError_t error, const std::string message) {
	if(error != cudaSuccess) {
		std::cout << message << " : " << cudaGetErrorString(error) << std::endl;
		system("PAUSE");
		std::exit(-1);
	}
}

static void initDevice() {
	// Set Device Configuration
	check(cudaSetDeviceFlags(cudaDeviceScheduleYield),"Change Schedule");
}

static int loadData(std::string &filename) {

	int dimension = -1;
	float *data;
	std::ifstream infile(filename);
	if(!infile.good()) {
		return -1;
	}
	
	int count = 0;
	
	std::string line;
	bool data_segment = false;
	while(std::getline(infile,line)) {
		std::istringstream iss(line);
		if(data_segment) {
			int i;
			float x,y;
			if(!(iss >> i >> x >> y)) {
				delete data;
				return -1;
			}
			data[count++] = x;
			data[count++] = y;
			if(count == dimension * 2) break;
		} else {
			std::string command;
			if(!(iss >> command)) return -1;
			if(command.find("DIMENSION") != std::string::npos) {
				iss >> line >> dimension;
				check(cudaMallocHost(&data,sizeof(float)*2*dimension),"cudaMallocHost");
			}else if(command.find("NODE_COORD_SECTION") != std::string::npos) {
				data_segment = true;
			}
		}
	}
	
	check(cudaGetLastError(),"After load");
	
	check(cudaMalloc(&device_best,sizeof(int)),"best_d");
	check(cudaMalloc(&device_climbs,sizeof(int)),"climbs_d");
	check(cudaMalloc(&device_pointer, 2 * dimension * sizeof(float)),"device_pointer");
	
	int best = INT_MAX;
	int climbs = 0;
	
	check(cudaMemcpy(device_pointer,data,2 * dimension * sizeof(float),cudaMemcpyHostToDevice),"Data : host->device");
	check(cudaMemcpy(device_best,&best,sizeof(int),cudaMemcpyHostToDevice),"Best : host->device");
	check(cudaMemcpy(device_climbs,&climbs,sizeof(int),cudaMemcpyHostToDevice),"Climbs : host->device");
	
	cudaFreeHost(data);
	return dimension;
}

static float runKernel(const int swaps, const int restarts, const int cities) {
	cudaEvent_t begin,end;
	float time;
	
	
	// Device properties
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,0);
	
	
	// Compute block, thread count, and heap size
	
	
	const int Threads = min(MAX_THREADS,32 * ((cities + 31) / 32));
	const int MaxBlocks = props.multiProcessorCount * min(16,2048/Threads);
	const int Blocks = min(restarts,MaxBlocks);
	
	const int MemorySize = 2 * Blocks * (cities + 1) * (2*sizeof(float) + sizeof(int));
	check(cudaDeviceSetLimit(cudaLimitMallocHeapSize, MemorySize),"Change heap size");
	
	// Info
	std::cout << "Blocks=" << Blocks << ", Threads=" << Threads << std::endl;
	std::cout << "Memory Size = " << MemorySize << std::endl;
	
	
	check(cudaMemcpyToSymbol(swaps_d,&swaps,sizeof(int)),"Swap : host->device");
	
	// Init events
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	// Run
	cudaEventRecord(begin,0);
		IterativeTwoOpt<<<Blocks,Threads>>>(device_pointer,cities,restarts,device_climbs,device_best);	
	cudaEventRecord(end,0);
	
	// Get time
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time,begin,end);
	check(cudaGetLastError(),"Kernel");
	
	// Cleanup
	cudaEventDestroy(end);
	cudaEventDestroy(begin);
	
	return time;
}

// Return the best path length
static int getTourLength() {
	cudaDeviceSynchronize();
	int best = 0;
	check(cudaMemcpy(&best,device_best,sizeof(int),cudaMemcpyDeviceToHost),"device->host");
	return best;
}

static int getClimbs() {
	int climbs  = 0;
	check(cudaMemcpy(&climbs,device_climbs,sizeof(int),cudaMemcpyDeviceToHost),"device->host");
	return climbs;
}

static void cleanup() {
	if(device_best!=NULL) cudaFree(device_best);
	if(device_climbs!=NULL) cudaFree(device_climbs);
}

int main(int argc, char** argv) {

	initDevice();
	
	// Error checking
	if(argc != 4) {
		std::cout << argv[0] << " <filename> <restarts> <swaps>" << std::endl;
		system("PAUSE");
		return EXIT_FAILURE;
	}

	std::string filename(argv[1]);
	std::string s_restarts(argv[2]);
	std::string s_swaps(argv[3]);
	
	const int cities = loadData(filename);
	
	if(cities == -1) {
		std::cout << "Could not read in data file" << std::endl;
		system("PAUSE");
		return EXIT_FAILURE;
	}
	
	try{
		const int restarts = std::stoi(s_restarts);
		const int swaps = min(cities,std::stoi(s_swaps));
		const float time = runKernel(swaps,restarts,cities);
		const int best = getTourLength();
		const int climbs = getClimbs();
		
		int hours = time / (3600.0f * 1000.0f);
		int seconds = (int)(time/1000) % 60;
		int minutes = (int)(time/1000) / 60;
		
		const long long moves = 1LL * climbs * (cities - 2) * (cities - 1) / 2;
		
		
		std::cout << moves * 0.000001 / time << "Gmoves/s" << std::endl;
		std::cout << "best found tour length = " << best << std::endl;
		std::cout << "Total Time : " << time / 1000.0f << "s" << std::endl;
		std::cout << "Hours = " << hours << ", Minutes = " << minutes << ", Seconds = " << seconds << ", Milliseconds = " << (int)(time) % 1000 << std::endl;
		
	}catch(std::exception& e) {
		std::cout << e.what() << std::endl;
		cleanup();
		cudaDeviceReset();
		system("PAUSE");
		return EXIT_FAILURE;
	}
	
	cleanup();
	cudaDeviceReset();
	return EXIT_SUCCESS;
}


