

// C++
#include <iostream>
#include <string>

// C
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>

// CUDA
#include <cuda.h>
#include <curand_kernel.h>

// Force -Wall after this point, VC only (Check https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html for GCC)
#pragma warning(push,4)

/******************************************************************************/
/*** 2-opt with random restarts ***********************************************/
/******************************************************************************/

#define dist(a, b) __float2int_rn(sqrtf((pos[a].x - pos[b].x) * (pos[a].x - pos[b].x) + (pos[a].y - pos[b].y) * (pos[a].y - pos[b].y)))
#define swap(a, b) {float tmp = a;  a = b;  b = tmp;}

static __device__ __managed__ int climbs_d = 0;
static __device__ __managed__ int best_d = INT_MAX;
static __device__ int restart_d = 0;
// Buffer space, used for cache and maximum propagation

enum ThreadBufferStatus {MORE_THREADS_THAN_BUFFER,EQUAL_SIZE,MORE_BUFFER_THAN_THREADS};

static __device__ __managed__ unsigned long long int d_lDuration = 0;
static __device__ __managed__ unsigned long long int d_cDuration = 0;
static __device__ __managed__ unsigned long long int d_uDuration = 0;
static __device__ __managed__ unsigned long long int d_sDuration = 0;

static __device__ long long int load_duration[128] = {0};
static __device__ long long int compute_duration[128] = {0};
static __device__ long long int update_duration[128] = {0};
static __device__ long long int single_duration[128] = {0};

#define DEBUG 0
#define MICRO 0
// Insturnmentation

// Load
static __device__ void inline load_start() {if(DEBUG && MICRO && threadIdx.x == 0) {load_duration[blockIdx.x] -= clock64();}}
static __device__ void inline load_end() {if(DEBUG && MICRO && threadIdx.x == 0) {load_duration[blockIdx.x] += clock64();}}

// Compute
static __device__ void inline compute_start() {if(DEBUG && MICRO && threadIdx.x == 0) {compute_duration[blockIdx.x] -= clock64();}}
static __device__ void inline compute_end() {if(DEBUG && MICRO && threadIdx.x == 0) {compute_duration[blockIdx.x] += clock64();}}

// Single_iter
static __device__ void inline single_start() {if(threadIdx.x == 0 && DEBUG) {single_duration[blockIdx.x] -= clock64();}}
static __device__ void inline single_end() {if(threadIdx.x == 0 && DEBUG) {single_duration[blockIdx.x] += clock64();}}

// Update
static __device__ void inline update_start() {if(DEBUG && threadIdx.x == 0) {update_duration[blockIdx.x] -= clock64();}}
static __device__ void inline update_end() {if(DEBUG && threadIdx.x == 0) {update_duration[blockIdx.x] += clock64();}}


#define GLOBALS(THS,BLK)													\
template <ThreadBufferStatus Status, int TileSize> 							\
static __global__ __launch_bounds__(THS, BLK ) void 						\
TwoOpt##THS(const int r, const Data *p, const int c) { 						\
	load_duration[blockIdx.x] = 0;											\
	compute_duration[blockIdx.x] = 0;										\
	TwoOpt<Status,TileSize>(r,p,c); 										\
	if(DEBUG && threadIdx.x==0) {											\
		atomicAdd(&d_lDuration,load_duration[blockIdx.x]);					\
		atomicAdd(&d_cDuration,compute_duration[blockIdx.x]);				\
		atomicAdd(&d_uDuration,update_duration[blockIdx.x]);				\
		atomicAdd(&d_sDuration,single_duration[blockIdx.x]);				\
	}																		\
}


// Data structure used to hold position along path
struct __align__(8) Data {
	float x,y;
};

//
// Returns a unique integer value with the intial value being 0
//
// @return  - Returns the next unique int
static __device__ inline int
nextInt(int* w_buffer) {
	if(threadIdx.x==0) {
		w_buffer[0] = atomicAdd(&restart_d,10);
	}__syncthreads();
	return w_buffer[0];
}

// Allocates and initializes my global memory and shared memory.
//
//	@pos	- An array that need to be initialized and will hold our path points
//	@weight	- An array that need to be initialized and will hold our edge weights
//	@cities	- The amount of points in our graph
//
//	@return	- Returns true if initialization was successful, false otherwise.
template <int TileSize>
static inline __device__ bool
initMemory(const Data* &pos_d, Data* &pos, int * &weight, const int &cities) {
	__shared__ Data *d;
	__shared__ int *w;
	// Allocate my global memory
	if(threadIdx.x == 0 ) {
		d = new Data[cities + 1];
		if(d != NULL) {
			w = new int[cities];
			if(w == NULL) {
				printf("Could not allocated for weight");
				delete d;
				d = NULL;
			}
		}else{
			printf("Could not allocate for position");
		}
	}__syncthreads();
	
	if(d == NULL) {
		return false;
	}
	
	// Save new memory locations
	pos = d;
	weight = w;
	
	for (int i = threadIdx.x; i < cities; i += blockDim.x) pos[i] = pos_d[i];
	__syncthreads();
	return true;
}

//
// Each thread gives some integer value, then the maximum of them is returned.
//
// @t_val  - The number that the thread submits as a candidate for the maximum value
// @cities - The number of cities.
//
// @return - The maximum value of t_val seen from all threads
template <ThreadBufferStatus Status, int TileSize>
static inline __device__ int
maximum(int t_val, const int &cities, int* __restrict__ &w_buffer) {
	
	int upper = min(blockDim.x,min(TileSize,cities));
	
	if(Status == MORE_THREADS_THAN_BUFFER) {
		const int Index = threadIdx.x % TileSize;
		w_buffer[Index] = t_val;
		__syncthreads();
		for(int i = 0 ; i <= (blockDim.x/TileSize); ++i ) {
			if(t_val < w_buffer[Index]) {
				w_buffer[Index] = t_val;
			}
		}
	}else {
		w_buffer[threadIdx.x] = t_val;
	}__syncthreads();
	
	// No
	if (TileSize > 512 && blockDim.x > 512) {
		int offset = (upper + 1) / 2;	// 200
		if( threadIdx.x < offset) {
			int tmp = w_buffer[threadIdx.x + offset];
			if(tmp < t_val) {
				w_buffer[threadIdx.x] = t_val = tmp;
			}
		}__syncthreads();
		upper = offset;
	}
	
	// No
	if (TileSize > 256 && blockDim.x > 256) {
		int offset = (upper + 1) / 2; // 100
		if( threadIdx.x < offset) {
			int tmp = w_buffer[threadIdx.x + offset];
			if(tmp < t_val) {
				w_buffer[threadIdx.x] = t_val = tmp;
			}
		}__syncthreads();
		upper = offset;
	}
	
	// No
	if (TileSize > 128 && blockDim.x > 128) {
		int offset = (upper + 1) / 2; // 50
		if( threadIdx.x < offset) {
			int tmp = w_buffer[threadIdx.x + offset];
			if(tmp < t_val) {
				w_buffer[threadIdx.x] = t_val = tmp;
			}
		}__syncthreads();
		upper = offset;
	}
	
	// No
	if (TileSize > 64 && blockDim.x > 64) {
		int offset = (upper + 1) / 2; // 25
		if( threadIdx.x < offset) {
			int tmp = w_buffer[threadIdx.x + offset];
			if(tmp < t_val) {
				w_buffer[threadIdx.x] = t_val = tmp;
			}
		}__syncthreads();
		upper = offset;
	}
	
	// 64 and down
	if(threadIdx.x < 32) {
		// Yes.  upper = 32.  w_buffer[tid] = t_val = min(t_val,w_buffer[threadIdx.x + 16]
		if(TileSize > 32 && blockDim.x > 32) {
			int tmp = w_buffer[threadIdx.x + (upper+1)/2];
			if(tmp < t_val) {
				w_buffer[threadIdx.x] = t_val = tmp;
			}
		}
		if(threadIdx.x < 16) {
			int tmp = w_buffer[threadIdx.x + 16];
			if(tmp < t_val) {
				w_buffer[threadIdx.x] = t_val = tmp;
			}
		}
		if(threadIdx.x < 8) {
			int tmp = w_buffer[threadIdx.x + 8];
			if(tmp < t_val) {
				w_buffer[threadIdx.x] = t_val = tmp;
			}
		}
		if(threadIdx.x < 4) {
			int tmp = w_buffer[threadIdx.x + 4];
			if(tmp < t_val) {
				w_buffer[threadIdx.x] = t_val = tmp;
			}
		}
		if(threadIdx.x < 2) {
			int tmp = w_buffer[threadIdx.x + 2];
			if(tmp < t_val) {
				w_buffer[threadIdx.x] = t_val = tmp;
			}
		}
		if(threadIdx.x < 1) {
			int tmp = w_buffer[threadIdx.x + 1];
			if(tmp < t_val) {
				w_buffer[threadIdx.x] = t_val = tmp;
			}
		}
	}__syncthreads();
	
	return w_buffer[0];
}

//
//	After we find the best four position to reconnect with all we need to 
//	reverse the path between them.
//
//	@start 	 - The first position in the sub-path we have to swap with the end
// 	@end	 - The last position in the path we have to swap with the start
//	@pos	 - The positions in our path
//	@weights - The edge weights between points
static inline __device__ void
reverse(int start, int end, Data* __restrict__ &pos, int* __restrict__ &weight) {
	while(start<end) {
	
		int   w = weight[start];
		Data d = pos[start];
		
		weight[start] = weight[end-1];
		pos[start] = pos[end];
		
		weight[end-1] = w;
		pos[end] = d;
		
		start += blockDim.x;
		end -= blockDim.x;
		
	}__syncthreads();
}

//
// Perform a single iteration of Two-Opt
// @pos			- The current Hamiltonian path 
// @weight		- The current weight of our edges along the path
// @minchange 	- The current best change we can make
// @mini		- The ith city in the path that is part of the swap
// @minj		- The jth city in the path that is part of the swap
// @cities		- The number of cities along the path (excluding the end point)
template <int TileSize>
static __device__ void
singleIter(Data* &pos, int* &weight, int &minchange,
	int &mini, int &minj, const int &cities,
	float* __restrict__ x_buffer, float* __restrict__ y_buffer, int* __restrict__ w_buffer) {
	single_start();
	for (int ii = 0; ii < cities - 2; ii += blockDim.x) {
		int i = ii + threadIdx.x;
		float pxi0, pyi0, pxi1, pyi1, pxj1, pyj1;
		
		if (i < cities - 2) {
			minchange -= weight[i];
			pxi0 = pos[i].x;
			pyi0 = pos[i].y;
			pxi1 = pos[i+1].x;
			pyi1 = pos[i+1].y;
			pxj1 = pos[0].x;
			pyj1 = pos[0].y;
		}
		
		for (int jj = cities - 1; jj >= ii + 2; jj -= TileSize) {
		
			int bound = jj - TileSize + 1;
			load_start();
			for(int k = threadIdx.x; k < TileSize; k += blockDim.x) {
				int index = k + bound;
				if (index >= (ii + 2)) {
					x_buffer[k] = pos[index].x;
					y_buffer[k] = pos[index].y;
					w_buffer[k] = weight[index];
				}
			}
			__syncthreads();
			load_end();
			compute_start();
			int lower = bound;
			if (lower < i + 2) lower = i + 2;
			
			for (int j = jj; j >= lower; j--) {
				int jm = j - bound;
				
				float pxj0 = x_buffer[jm];
				float pyj0 = y_buffer[jm];
				int change = w_buffer[jm]
				
					+ __float2int_rn(sqrtf((pxi0 - pxj0) * (pxi0 - pxj0) + (pyi0 - pyj0) * (pyi0 - pyj0)))
					+ __float2int_rn(sqrtf((pxi1 - pxj1) * (pxi1 - pxj1) + (pyi1 - pyj1) * (pyi1 - pyj1)));
					
				pxj1 = pxj0;
				pyj1 = pyj0;
				
				if (minchange > change) {
					minchange = change;
					mini = i;
					minj = j;
				}
			}
			__syncthreads();
			compute_end();
			
		}

		if (i < cities - 2) {
			minchange += weight[i];
		}
	}
	single_end();
}

//
// Perform the swaps to the edges i and j to decrease the total length of our 
// path and update the weight and pos arrays appropriately.
//
// @pos			- The current Hamiltonian path 
// @weight		- The current weight of our edges along the path
// @minchange 	- The current best change we can make
// @mini		- The ith city in the path that is part of the swap
// @minj		- The jth city in the path that is part of the swap
// @cities		- The number of cities along the path (excluding the end point)
template <ThreadBufferStatus Status, int TileSize>
static __device__ bool
update(Data* &pos, int* &weight, int &minchange, int &mini, int &minj, const int &cities, int* __restrict__ w_buffer) {
	update_start();
	maximum<Status,TileSize>(minchange, cities, w_buffer);
	if(w_buffer[0] >= 0) {
		update_end();
		return false;
	}
	
	while(w_buffer[0] < 0) {
	
		if (minchange == w_buffer[0]) {
			w_buffer[1] = threadIdx.x;
		}__syncthreads();
		
		if(threadIdx.x == w_buffer[1]) {
			w_buffer[2] = mini;
			w_buffer[3] = minj;
		}__syncthreads();
		
		int mi = w_buffer[2];
		int mj = w_buffer[3];
		
		if(!(minj < (mi - 1)) && !(mini > (mj + 1))) {
			minchange = 0;
		}
		
		// Fix path and weights
		reverse(mi+1+threadIdx.x,mj-threadIdx.x,pos,weight);
		
		// Fix connecting points
		weight[mi] = -dist(mi,mi+1);
		weight[mj] = -dist(mj,mj+1);
		__syncthreads();
		maximum<Status,TileSize>(minchange, cities, w_buffer);
	}
	update_end();
	return true;
}

//
// Given a path we randomly permute it into a new new path and then initialize the weights of the path.
//
// @pos			- The current Hamiltonian path 
// @weight		- The current weight of our edges along the path
// @cities		- The number of cities along the path (excluding the end point)
static __device__ inline void
permute(Data* &pos, int* &weight, const int &cities) {
	if (threadIdx.x == 0) {  // serial permutation
		curandState rndstate;
		curand_init(blockIdx.x, 0, 0, &rndstate);
		for (int i = 1; i < cities; i++) {
			int j = curand(&rndstate) % (cities - 1) + 1;
			
			Data d = pos[i];
			pos[i] = pos[j];
			pos[j] = d;
		}
		pos[cities] = pos[0];
	}__syncthreads();
	
	for (int i = threadIdx.x; i < cities; i += blockDim.x) weight[i] = -dist(i, i + 1);
	__syncthreads();
	
}



//
// Releases memory and saves results
//
// @pos				- Pointer to allocated path memory
// @weight			- Pointer to allocated edge weight memory
// @local_climbs	- The number of climbs performed by this block
// @best_length		- The best length this block found.
static __device__ inline void cleanup(Data* &pos, int* &weight, int &local_climbs, int &best_length) {
	if (threadIdx.x == 0) {
		// Save data
		atomicAdd(&climbs_d,local_climbs);
		atomicMin(&best_d, best_length);
		
		// Release memory
		delete pos;
		delete weight;
	}
}


//
// Perform iterative two-opt until there can be no more swaps to reduce the path length.
//
// @pos_d	- The position of each point in the graph.
// @cities	- The number of vertices in the graph
template <ThreadBufferStatus Status, int TileSize>
static __device__ void
TwoOpt(const int Restarts, const Data *pos_d, const int cities) {
	
	__shared__ float x_buffer[TileSize];
	__shared__ float y_buffer[TileSize];
	__shared__ int w_buffer[TileSize];

	Data	*pos;
	int 	*weight;
	int 	local_climbs = 0;
	int		best_length = INT_MAX;	// I really only need this for one thread.
	
	best_length = INT_MAX;
	
	if( !initMemory<TileSize>(pos_d,pos,weight,cities) ) {
		if(threadIdx.x == 0) {
			printf("Memory initialization error for block %d\n", blockIdx.x);
		}
		return;
	}
	
	for(int r = nextInt(w_buffer) ; r < Restarts; r = nextInt(w_buffer)) {
		int test = min(Restarts - r, 10);
		for(int i = 0 ; i < test ; ++i ) {
			int mini,minj,minchange;
			
			permute(pos,weight,cities);
		  
			do {
				++local_climbs;
				minchange = mini = minj = 0;
				singleIter<TileSize>(pos, weight, minchange, mini, minj, cities, x_buffer, y_buffer, w_buffer);
			} while (update<Status,TileSize>(pos, weight, minchange, mini, minj, cities, w_buffer));
		
			w_buffer[0] = 0;
			__syncthreads();
			mini = 0;
			for (int i = threadIdx.x; i < cities; i += blockDim.x) {
				mini += dist(i, i + 1);
			}
			atomicAdd(&w_buffer[0],mini);
			__syncthreads();

			if(threadIdx.x==0) {
				if(w_buffer[0] < best_length) {
					best_length = w_buffer[0];
				}
			}
		}
	}
	cleanup(pos, weight, local_climbs, best_length);
}

// I hate this so much
GLOBALS(128,16)
GLOBALS(160,12)
GLOBALS(192,10)
GLOBALS(224,9)
GLOBALS(256,8)
GLOBALS(288,7)
GLOBALS(320,6)
GLOBALS(384,5)
GLOBALS(512,4)
GLOBALS(672,3)
GLOBALS(1024,2)


//
// Checks to see if an error occured with CUDA and if so prints out the message passed and the CUDA 
// error then quits the application.
//
// @msg	- Message to print out if error occurs
static void
CudaTest(char *msg) {
  cudaError_t e;
  cudaThreadSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
	system("PAUSE");
    exit(-1);
  }
}

#define mallocOnGPU(addr, size) if (cudaSuccess != cudaMalloc((void **)&addr, size)) fprintf(stderr, "could not allocate GPU memory\n");  CudaTest("couldn't allocate GPU memory");
#define copyToGPU(to, from, size) if (cudaSuccess != cudaMemcpy(to, from, size, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of data to device failed\n");  CudaTest("data copy to device failed");

//
// Read TPS lib files into GPU memory.  ATT and CEIL_2D edge weight types are not supported
//
// @fname	- The name of the file to read the TSP data from
// @pos_d	- Pointer to the pointer that will hold data on GPU
//			  and is modified here to be the address on the GPU
//
// @return	- Returns the number of cities found
static int
readInput(const char *fname, Data **pos_d) {
  int ch, cnt, in1, cities;
  float in2, in3;
  FILE *f;
  Data *pos;
  char str[256];  // potential for buffer overrun

  f = fopen(fname, "rt");
  if (f == NULL) {fprintf(stderr, "could not open file %s\n", fname);  exit(-1);}

  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);

  ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
  fscanf(f, "%s\n", str);
  cities = atoi(str);
  if (cities <= 2) {fprintf(stderr, "only %d cities\n", cities);  exit(-1);}

  pos = new Data[cities];  if (pos == NULL) {fprintf(stderr, "cannot allocate pos\n");  exit(-1);}

  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  fscanf(f, "%s\n", str);
  if (strcmp(str, "NODE_COORD_SECTION") != 0) {fprintf(stderr, "wrong file format\n");  exit(-1);}

  cnt = 0;
  while (fscanf(f, "%d %f %f\n", &in1, &in2, &in3)) {
  
    pos[cnt].x = in2;
    pos[cnt].y = in3;
	
	++cnt;
	
    if (cnt > cities) {fprintf(stderr, "input too long\n");  exit(-1);}
    if (cnt != in1) {fprintf(stderr, "input line mismatch: expected %d instead of %d\n", cnt, in1);  exit(-1);}
  }
  if (cnt != cities) {fprintf(stderr, "read %d instead of %d cities\n", cnt, cities);  exit(-1);}

  fscanf(f, "%s", str);
  if (strcmp(str, "EOF") != 0) {fprintf(stderr, "didn't see 'EOF' at end of file\n");  exit(-1);}

  mallocOnGPU(*pos_d, sizeof(Data) * cities);
  copyToGPU(*pos_d, pos, sizeof(Data) * cities);

  fclose(f);
  
  delete (pos);

  return cities;
}

//
// Given an enum value return it's string representation
//
// @status - The enum value to translate
//
// @return - The enums string representation in the source code
static const std::string
getName(const ThreadBufferStatus status) {
	switch(status) {
		case MORE_THREADS_THAN_BUFFER:
			return std::string("MORE_THREADS_THAN_BUFFER");
		case EQUAL_SIZE:
			return std::string("EQUAL_SIZE");
		case MORE_BUFFER_THAN_THREADS:
			return std::string("MORE_BUFFER_THAN_THREADS");
	};
	return std::string("enum value not found.");
}


//
// Calculates the maximum number of resident blocks that the card can hold
//
// @Threads 		- Number of threads that each block will have
// @Shared_Bytes	- The amount of bytes each block will allocate
//
// @return 			- Returns the number of blocks the card can have resident
static int
getMaxBlocks(const int Shared_Bytes, const int Threads) {
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,0);

	if(props.major < 3) {
		const int Max_Shared = 16384;
		const int Block_Shared_Limit = (Max_Shared / Shared_Bytes);
		return props.multiProcessorCount * min(8,min(Block_Shared_Limit,(int)(2048/Threads)));
	}else if(props.major < 5) {
		const int Max_Shared = 32768;
		const int Block_Shared_Limit = (Max_Shared / Shared_Bytes);
		return props.multiProcessorCount * min(16,min(Block_Shared_Limit,(int)(2048/Threads)));
	}else {
		const int Max_Shared = 65536;
		const int Block_Shared_Limit = (Max_Shared / Shared_Bytes);
		return props.multiProcessorCount * min(32,min(Block_Shared_Limit,(int)(2048/Threads)));
	}
}


//
// Given an integer returns the next multiple of 32 greater than or equal to it.
//
// @in 		- The integer to round to next multiple of 32
//
// @return 	- Returns the next multiple of 32 that is greater than or equals to in
static int
next32(int in) {
	return ((in + 31) / 32 ) * 32;
}

#define EXPAND(THRDS) \
	cudaEventRecord(begin,0);\
	TwoOpt##THRDS<Status,TileSize><<<Blocks,Threads>>>(Restarts,Pos_d,Cities);\
	CudaTest("Kernel Call");\
	cudaEventRecord(end,0);\
	cudaEventSynchronize(end);

template <ThreadBufferStatus Status, int TileSize>
static float
_wrapThreads(const int Restarts, const int Blocks, const int Threads, const Data *Pos_d, const int Cities) {

	float time;
	cudaEvent_t begin,end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	if(Threads > 672) {
		EXPAND(1024);
	}else if(Threads > 512) {
		EXPAND(672);
	}else if(Threads > 384) {
		EXPAND(512);
	}else if(Threads > 320) {
		EXPAND(384);
	}else if(Threads > 288) {
		EXPAND(320);
	}else if(Threads > 256) {
		EXPAND(288);
	}else if(Threads > 224) {
		EXPAND(256);
	}else if(Threads > 192) {
		EXPAND(224);
	}else if(Threads > 160) {
		EXPAND(192);
	}else if(Threads > 128) {
		EXPAND(160);
	}else {
		EXPAND(128);
	}
	
	cudaEventElapsedTime(&time,begin,end);
	cudaEventDestroy(begin);
	cudaEventDestroy(end);
	
	return time;
}

//
// Handle ThreadBufferStatus kernel selection
//
template <int TileSize>
static float
_wrapStatus(const int Restarts, const int Threads, const Data *Pos_d, const int Cities) {

	const int Shared_Bytes = (sizeof(int) + 2*sizeof(float)) * TileSize;
	const int Blocks = min(Restarts,getMaxBlocks(Shared_Bytes,Threads));
	const ThreadBufferStatus Status = (Threads > TileSize) ? MORE_THREADS_THAN_BUFFER : (Threads < TileSize) ? MORE_BUFFER_THAN_THREADS : EQUAL_SIZE;

	const int DeviceBytes = (sizeof(int) + sizeof(Data)) * (Cities + 1)* 2 * Blocks;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, DeviceBytes);
	CudaTest("Change heap size");
	
	// Output runtime configuration
	std::cout << "Blocks = " << Blocks << ", Threads  = " << Threads << ", TileSize = " << TileSize << ", Status = " << getName(Status) << ", Shared Bytes = " << Shared_Bytes << ", Device Memory = " << DeviceBytes/(1024.0f*1024.0f) << "MB" << std::endl;

	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	switch(Status) {
		case MORE_THREADS_THAN_BUFFER:
			return _wrapThreads<MORE_THREADS_THAN_BUFFER,TileSize>(Restarts, Blocks, Threads, Pos_d, Cities);
		case EQUAL_SIZE:
			return _wrapThreads<EQUAL_SIZE,TileSize>(Restarts, Blocks, Threads, Pos_d, Cities);
		case MORE_BUFFER_THAN_THREADS:
			return _wrapThreads<MORE_BUFFER_THAN_THREADS,TileSize>(Restarts, Blocks, Threads, Pos_d, Cities);
	};
	

	return 0.0;
}


//
// Outer call to kernel selection
//
static float
RunKernel(const int Cities, const Data *Pos, const int Restarts, const int Threads, const int TileSize) {
	switch(TileSize) {
		case 32:
			return _wrapStatus<32>(Restarts,Threads,Pos,Cities);
		case 64:
			return _wrapStatus<64>(Restarts,Threads,Pos,Cities);
		case 96:
			return _wrapStatus<96>(Restarts,Threads,Pos,Cities);
		case 128:
			return _wrapStatus<128>(Restarts,Threads,Pos,Cities);
		case 160:
			return _wrapStatus<160>(Restarts,Threads,Pos,Cities);
		case 192:
			return _wrapStatus<192>(Restarts,Threads,Pos,Cities);
		case 224:
			return _wrapStatus<224>(Restarts,Threads,Pos,Cities);
		case 256:
			return _wrapStatus<256>(Restarts,Threads,Pos,Cities);
		case 288:
			return _wrapStatus<288>(Restarts,Threads,Pos,Cities);
		case 320:
			return _wrapStatus<320>(Restarts,Threads,Pos,Cities);
		case 352:
			return _wrapStatus<352>(Restarts,Threads,Pos,Cities);
		case 384:
			return _wrapStatus<384>(Restarts,Threads,Pos,Cities);
		case 416:
			return _wrapStatus<416>(Restarts,Threads,Pos,Cities);
		case 448:
			return _wrapStatus<448>(Restarts,Threads,Pos,Cities);
		case 480:
			return _wrapStatus<480>(Restarts,Threads,Pos,Cities);
		case 512:
			return _wrapStatus<512>(Restarts,Threads,Pos,Cities);
		case 544:
			return _wrapStatus<544>(Restarts,Threads,Pos,Cities);
		case 576:
			return _wrapStatus<576>(Restarts,Threads,Pos,Cities);
		case 608:
			return _wrapStatus<608>(Restarts,Threads,Pos,Cities);
		case 640:
			return _wrapStatus<640>(Restarts,Threads,Pos,Cities);
		case 672:
			return _wrapStatus<672>(Restarts,Threads,Pos,Cities);
		case 704:
			return _wrapStatus<704>(Restarts,Threads,Pos,Cities);
		case 736:
			return _wrapStatus<736>(Restarts,Threads,Pos,Cities);
		case 768:
			return _wrapStatus<768>(Restarts,Threads,Pos,Cities);
		case 800:
			return _wrapStatus<800>(Restarts,Threads,Pos,Cities);
		case 832:
			return _wrapStatus<832>(Restarts,Threads,Pos,Cities);
		case 864:
			return _wrapStatus<864>(Restarts,Threads,Pos,Cities);
		case 896:
			return _wrapStatus<896>(Restarts,Threads,Pos,Cities);
		case 928:
			return _wrapStatus<928>(Restarts,Threads,Pos,Cities);
		case 960:
			return _wrapStatus<960>(Restarts,Threads,Pos,Cities);
		case 992:
			return _wrapStatus<992>(Restarts,Threads,Pos,Cities);
		case 1024:
			return _wrapStatus<1024>(Restarts,Threads,Pos,Cities);
		default:
			std::cout << "Invalid TileSize = " << TileSize << std::endl;
			exit(-1);
	};
	return -1;
}


static int
ComputeThreads(const int Threads) {
	if(Threads > 672) {
		return 1024;
	}else if(Threads > 512) {
		return 672;
	}else if(Threads > 384) {
		return 512;
	}else if(Threads > 320) {
		return 384;
	}else if(Threads > 288) {
		return 320;
	}else if(Threads > 256) {
		return 288;
	}else if(Threads > 224) {
		return 256;
	}else if(Threads > 192) {
		return 224;
	}else if(Threads > 160) {
		return 192;
	}else if(Threads > 128) {
		return 160;
	}else {
		return 128;
	}
}

//
//	Main entry point to program.
//
int
main(int argc, char *argv[]) {

	if (argc < 3 || argc > 5) {fprintf(stderr, "\narguments: input_file restart_count <threads> <tilesize> \n"); exit(-1);}

	const int Restarts = atoi(argv[2]);
	if (Restarts < 1) {fprintf(stderr, "restart_count is too small: %d\n", Restarts); exit(-1);}

	Data *pos_d;
	const int Cities = readInput(argv[1], &pos_d);
	printf("configuration: %d cities, %d restarts, %s input\n", Cities, Restarts, argv[1]);


	const int Threads = ComputeThreads(32*(((argc >= 4) ? atoi(argv[3]) : Cities)/32));
	const int TileSize = (argc >= 5) ? min( next32(atoi(argv[4])),1024) : Threads;
	
	const float time = RunKernel(Cities,pos_d,Restarts,Threads,TileSize);
	
	int hours = (int)(time / (3600.0f * 1000.0f));
	int seconds = (int)(time/1000) % 60;
	int minutes = (int)(time/1000) / 60;
	
	
	long long moves = 1LL * climbs_d * (Cities - 2) * (Cities - 1) / 2;
	
	std::cout << moves * 0.000001 / time << "Gmoves/s" << std::endl;
	std::cout << "best found tour length = " << best_d << std::endl;
	std::cout << "Total Time : " << time / 1000.0f << "s" << std::endl;
	std::cout << "Hours = " << hours << ", Minutes = " << minutes << ", Seconds = " << seconds << ", Milliseconds = " << (int)(time) % 1000 << std::endl;
	#if DEBUG
	std::cout << "Load Duration: " << d_lDuration << ", Compute Duration: " << d_cDuration << std::endl;
	std::cout << "Single Duration: " << d_sDuration << ", Update Duration: " << d_uDuration << std::endl;
	#endif
	cudaDeviceReset();
	cudaFree(pos_d);
	return 0;
}

