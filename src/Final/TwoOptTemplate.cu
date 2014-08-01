
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
// #define TileSize 1024
//#define MAX_THREADS 1024

// CPU POINTERS TO GPU ADDRESSES
static float *device_pointer;
static int *device_climbs;
static int *device_best;

// GPU MEMORY ADDRESSES
static __device__ int *climbs_d;
static __device__ int *length_d;
static __device__ int index_d = 0;

// SHARED SCRATCH MEMORY
__shared__ float *x_buffer;
__shared__ float *y_buffer;
__shared__ int *w_buffer;


extern __shared__ char total_buffer[];

static __device__ int nextIndex() {
	
	if(threadIdx.x==0) {
		w_buffer[0] = atomicAdd(&index_d,1);
	}__syncthreads();
	return w_buffer[0];
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
template <int TileSize>
static __device__ bool initMemory(const float *data_d,float* &data, int* &weight,const int &cities) {	
	__shared__ float *t_f;
	__shared__ int *t_i;
	
	x_buffer = (float*)total_buffer;
	y_buffer = (float*)(total_buffer + sizeof(float) * TileSize);
	w_buffer = (int*)(total_buffer + 2 * sizeof(float) * TileSize);
		
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
template <int Threads, int TileSize>
static __device__ void loadShared(const float* data, const int* weight, const int &bound) {
	for(int i = threadIdx.x; i < TileSize; i += Threads) {
		int index = i + bound;
		if(index > 0) {
			x_buffer[i] = data[2 * index + 0];
			y_buffer[i] = data[2 * index + 1];
			w_buffer[i] = weight[index];
		}
		index -= TileSize;
		if(index > 0) {
			asm("prefetch.global.L1 [%0];" :: "l"(data + 2 * index + 0));
			asm("prefetch.global.L1 [%0];" :: "l"(data + 2 * index + 1));
			asm("prefetch.global.L1 [%0];" :: "l"(weight + index));
		}
	}__syncthreads();
}

// Perform a single iteration of 2opt
template <int Threads, int TileSize>
static __device__ void singleTwoOpt(const float* data, const int* weight, const int &cities, int &best, int &best_a, int &best_b) {

	for(int front = 0 ; front < cities - 2; front += Threads) {
	
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
			
			loadShared<Threads,TileSize>(data,weight,bound);
			
			int lower = max(bound,city+2);
			
			for(int j = back; j >= lower; --j) {
				int jm = j - bound;
				
				register float c3x = x_buffer[jm];
				register float c3y = y_buffer[jm];
				
				int candidate = d12 + w_buffer[jm] - (dist(c1x,c3x,c1y,c3y) + dist(c2x,c4x,c2y,c4y));
				
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


// Propagate maximum value to w_buffer[0]
template <int Threads,int TileSize>
static __device__ void maximum(int t_val, const int &cities) {

	int index;
	if(Threads>TileSize) {
		index = threadIdx.x % TileSize;
		w_buffer[index] = t_val;
		__syncthreads();
		
		for(int i = 1 ; i < (Threads / TileSize); ++i) {
			if(t_val > w_buffer[index]) {
				w_buffer[index] = t_val;
			}__syncthreads();
		}
		index = min(cities,TileSize);
	}else{
		w_buffer[threadIdx.x] = t_val;
		__syncthreads();
		index = min(cities,Threads);
	}
	
	
	#pragma unroll
	for(int i = 1 ; i < 11 ; ++i ) {
		index = (index + 1) / 2;
		if(threadIdx.x < index) {
			int tmp = w_buffer[threadIdx.x + index];
			if(t_val < tmp) {
				w_buffer[threadIdx.x] = t_val = tmp;
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
template <int K,int Threads, int TileSize>
static __device__ bool update(float* &data, int* &weight,const int &cities,int &best, int &front, int &back) {
	
	// Get maximum best from all threads
	maximum<Threads,TileSize>(best,cities);
	
	// If none then jump
	if(w_buffer[0] == 0) {
		return false;
	}
	
	
	#pragma unroll
	for(int i = 0 ; i < K; ++i) {
		// Get all the data needed
		if(w_buffer[0]==best) {
			w_buffer[1] = threadIdx.x;
		}__syncthreads();
		
		if(threadIdx.x==w_buffer[1]) {
			best = 0;
			w_buffer[2] = front;
			w_buffer[3] = back;
		}__syncthreads();
		
		if(!(back < w_buffer[2]-1) && !(front > w_buffer[3]+1)) {
			best = 0;
		}
		
		reverse(data,weight,w_buffer[2]+threadIdx.x+1,w_buffer[3]-threadIdx.x);
		
		// I can split this across two threads!
		if(threadIdx.x<2) {
			int s = w_buffer[2+threadIdx.x];
			weight[s] = distv(data,s,s+1);
		}__syncthreads();
		
		maximum<Threads,TileSize>(best,cities);
		
		// If none then jump
		if(w_buffer[0] == 0) {
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

template <int K, int Threads,int TileSize>
static __device__ void IterativeTwoOpt_d(const float* data_d, const int cities,const int restarts, int *climbs, int *len) {
	// save to global
	length_d = len;
	climbs_d = climbs;
	
	float* data;
	int *weight;
	int local_climbs = 0;
	int length = INT_MAX;
	
	// Create memory for block
	if(!initMemory<TileSize>(data_d,data,weight,cities)) {
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
			singleTwoOpt<Threads,TileSize>(data,weight,cities,best,front,back);
			++local_climbs;
		}while(update<K,Threads,TileSize>(data,weight,cities,best,front,back));
		
		// Calculate the path
		w_buffer[0] = 0;
		__syncthreads();
		
		best = 0;
		for(int i = threadIdx.x ; i < cities; i+= blockDim.x) {
			best += weight[i];
		}atomicAdd(w_buffer,best);
		__syncthreads();
		
		// Save if better
		if( threadIdx.x==0 && w_buffer[0] < length) {
			length  = w_buffer[0];
		}
	}
	
	// finally done so save our best ones
	saveResults(local_climbs,length);	
	releaseMemory(data,weight);
}

template <int K, int Threads,int TileSize>
static __global__ void __launch_bounds__(1024,2) IterativeTwoOpt(const float* data_d, const int cities,const int restarts, int *climbs, int *len) {
	IterativeTwoOpt_d<K,Threads,TileSize>(data_d,cities,restarts,climbs,len);
}

template <int K, int Threads,int TileSize>
static __global__ void __launch_bounds__(32,2) IterativeTwoOpt32(const float* data_d, const int cities,const int restarts, int *climbs, int *len) {
	IterativeTwoOpt_d<K,Threads,TileSize>(data_d,cities,restarts,climbs,len);
}

template <int K, int Threads,int TileSize>
static __global__ void __launch_bounds__(64,2) IterativeTwoOpt64(const float* data_d, const int cities,const int restarts, int *climbs, int *len) {
	IterativeTwoOpt_d<K,Threads,TileSize>(data_d,cities,restarts,climbs,len);
}


template <int K, int Threads,int TileSize>
static __global__ void __launch_bounds__(128,2) IterativeTwoOpt128(const float* data_d, const int cities,const int restarts, int *climbs, int *len) {
	IterativeTwoOpt_d<K,Threads,TileSize>(data_d,cities,restarts,climbs,len);
}

template <int K, int Threads,int TileSize>
static __global__ void __launch_bounds__(256,2) IterativeTwoOpt256(const float* data_d, const int cities,const int restarts, int *climbs, int *len) {
	IterativeTwoOpt_d<K,Threads,TileSize>(data_d,cities,restarts,climbs,len);
}

template <int K, int Threads,int TileSize>
static __global__ void __launch_bounds__(512,2) IterativeTwoOpt512(const float* data_d, const int cities,const int restarts, int *climbs, int *len) {
	IterativeTwoOpt_d<K,Threads,TileSize>(data_d,cities,restarts,climbs,len);
}

template <int K, int Threads,int TileSize>
static __global__ void __launch_bounds__(1024,2) IterativeTwoOpt1024(const float* data_d, const int cities,const int restarts, int *climbs, int *len) {
	IterativeTwoOpt_d<K,Threads,TileSize>(data_d,cities,restarts,climbs,len);
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

template <int K, int TileSize>
static float runKernel(const int restarts, const int cities) {
	cudaEvent_t begin,end;
	float time;
	
	// Device properties
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,0);
	
	// Compute block, thread count, and heap size
	
	const int Threads = min(1024,32 * ((cities + 31) / 32));
	const int MaxBlocks = props.multiProcessorCount * min(16,2048/Threads);
	const int Blocks = min(restarts,MaxBlocks);
	const int MemorySize = 2 * Blocks * (cities + 1) * (2*sizeof(float) + sizeof(int));
	check(cudaDeviceSetLimit(cudaLimitMallocHeapSize, MemorySize),"Change heap size");
	
	// Info
	std::cout << "Swaps=" << K << std::endl;
	std::cout << "TileSize=" << TileSize << std::endl;
	std::cout << "Blocks=" << Blocks << ", Threads=" << Threads << std::endl;
	std::cout << "Memory Size = " << MemorySize << std::endl;
	
	// Init events
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	const int SwapBytes = TileSize * (sizeof(int) + 2*sizeof(float));
	
	// Run
	cudaEventRecord(begin,0);
	switch(Threads) {
		case 32:
			IterativeTwoOpt32<K,32,TileSize><<<Blocks,32,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 64:
			IterativeTwoOpt64<K,64,TileSize><<<Blocks,64,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 96:
			IterativeTwoOpt<K,96,TileSize><<<Blocks,96,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 128:
			IterativeTwoOpt128<K,128,TileSize><<<Blocks,128,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 160:
			IterativeTwoOpt<K,160,TileSize><<<Blocks,160,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 192:
			IterativeTwoOpt<K,192,TileSize><<<Blocks,192,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 224:
			IterativeTwoOpt<K,224,TileSize><<<Blocks,224,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 256:
			IterativeTwoOpt256<K,256,TileSize><<<Blocks,256,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 288:
			IterativeTwoOpt<K,288,TileSize><<<Blocks,288,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 320:
			IterativeTwoOpt<K,320,TileSize><<<Blocks,320,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 352:
			IterativeTwoOpt<K,352,TileSize><<<Blocks,352,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 384:
			IterativeTwoOpt<K,384,TileSize><<<Blocks,384,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 416:
			IterativeTwoOpt<K,416,TileSize><<<Blocks,416,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 448:
			IterativeTwoOpt<K,448,TileSize><<<Blocks,448,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 480:
			IterativeTwoOpt<K,480,TileSize><<<Blocks,480,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 512:
			IterativeTwoOpt512<K,512,TileSize><<<Blocks,512,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 544:
			IterativeTwoOpt<K,544,TileSize><<<Blocks,544,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 576:
			IterativeTwoOpt<K,576,TileSize><<<Blocks,576,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 608:
			IterativeTwoOpt<K,608,TileSize><<<Blocks,608,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 640:
			IterativeTwoOpt<K,640,TileSize><<<Blocks,640,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 672:
			IterativeTwoOpt<K,672,TileSize><<<Blocks,672,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 704:
			IterativeTwoOpt<K,704,TileSize><<<Blocks,704,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 736:
			IterativeTwoOpt<K,736,TileSize><<<Blocks,736,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 768:
			IterativeTwoOpt<K,768,TileSize><<<Blocks,768,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 800:
			IterativeTwoOpt<K,800,TileSize><<<Blocks,800>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 832:
			IterativeTwoOpt<K,832,TileSize><<<Blocks,832,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 864:
			IterativeTwoOpt<K,864,TileSize><<<Blocks,864,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 896:
			IterativeTwoOpt<K,896,TileSize><<<Blocks,896,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 928:
			IterativeTwoOpt<K,928,TileSize><<<Blocks,928,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 960:
			IterativeTwoOpt<K,960,TileSize><<<Blocks,960,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 992:
			IterativeTwoOpt<K,992,TileSize><<<Blocks,992,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
		case 1024:
			IterativeTwoOpt1024<K,1024,TileSize><<<Blocks,1024,SwapBytes>>>(device_pointer,cities,restarts,device_climbs,device_best);
			break;
	}
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

static int pow2(int n) {
	--n;
	n |= n >> 1;
	n |= n >> 2;
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	++n;
	return n;
}

static void cleanup() {
	if(device_best!=NULL) cudaFree(device_best);
	if(device_climbs!=NULL) cudaFree(device_climbs);
}

template <int Swaps>
static int swapWrapper(int TileSize, int restarts, int cities) {
	switch(TileSize) {
		case 32:
			return runKernel<Swaps,32>(restarts,cities);
		case 64:
			return runKernel<Swaps,64>(restarts,cities);
		case 128:
			return runKernel<Swaps,128>(restarts,cities);
		case 256:
			return runKernel<Swaps,256>(restarts,cities);
		case 512:
			return runKernel<Swaps,512>(restarts,cities);
		case 1024:
			return runKernel<Swaps,1024>(restarts,cities);
		default:
			return runKernel<Swaps,128>(restarts,cities);
	}
}

int main(int argc, char** argv) {

	initDevice();

	// Error checking
	if(argc != 5) {
		std::cout << argv[0] << " <filename> <restarts> <swaps> <tilesize>" << std::endl;
		system("PAUSE");
		return EXIT_FAILURE;
	}

	std::string filename(argv[1]);
	std::string s_restarts(argv[2]);
	std::string s_swaps(argv[3]);
	std::string s_tilesize(argv[4]);
	
	
	
	const int cities = loadData(filename);
	
	
	if(cities == -1) {
		std::cout << "Could not read in data file" << std::endl;
		system("PAUSE");
		return EXIT_FAILURE;
	}
	
	try{
		const int Restarts = std::stoi(s_restarts);
		const int Swaps = pow2(min(cities,std::stoi(s_swaps)));
		const int TileSize = pow2(min(cities,std::stoi(s_tilesize)));
		
		float time;
		switch(Swaps) {
			case 1:
				time = swapWrapper<1>(TileSize,Restarts,cities) / 1000.0f;	
				break;
			case 2:
				time = swapWrapper<2>(TileSize,Restarts,cities) / 1000.0f;	
				break;
			case 4:
				time = swapWrapper<4>(TileSize,Restarts,cities) / 1000.0f;	
				break;
			case 8:
				time = swapWrapper<8>(TileSize,Restarts,cities) / 1000.0f;	
				break;
			case 16:
				time = swapWrapper<16>(TileSize,Restarts,cities) / 1000.0f;	
				break;
			case 32:
				time = swapWrapper<32>(TileSize,Restarts,cities) / 1000.0f;	
				break;
			case 64:
				time = swapWrapper<64>(TileSize,Restarts,cities) / 1000.0f;	
				break;
			case 128:
				time = swapWrapper<128>(TileSize,Restarts,cities) / 1000.0f;	
				break;
			case 256:
				time = swapWrapper<256>(TileSize,Restarts,cities) / 1000.0f;	
				break;
			case 512:
				time = swapWrapper<512>(TileSize,Restarts,cities) / 1000.0f;	
				break;
			case 1024:
				time = swapWrapper<1024>(TileSize,Restarts,cities) / 1000.0f;	
				break;
			case 2048:
				time = swapWrapper<2048>(TileSize,Restarts,cities) / 1000.0f;	
				break;
		}
		
		const int best = getTourLength();
		const int climbs = getClimbs();
		
		const long long moves = 1LL * climbs * (cities - 2) * (cities - 1) / 2;
		std::cout << "Time : " << time << "s, Length = " << best <<  std::endl;
		std::cout << moves * 0.000000001 / time << " GMoves/s " << std::endl;
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


