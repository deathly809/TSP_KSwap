
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
#include <cuda_profiler_api.h>

// Force -Wall after this point, VC only (Check https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html for GCC)
#pragma warning(push,4)

#define DEBUG 0
#define MICRO 0
#define MACRO 0

#define QUEUE 0



#if QUEUE
const int MaxBlocks = 10000;
const int SliceSize = 100;
#endif


/******************************************************************************/
/*** 2-opt with random restarts ***********************************************/
/******************************************************************************/

// Euclidean distance
#define dist(a, b) __float2int_rn(sqrtf((pos[a].x - pos[b].x) * (pos[a].x - pos[b].x) + (pos[a].y - pos[b].y) * (pos[a].y - pos[b].y)))
#define swap(a, b) {float tmp = a;  a = b;  b = tmp;}

static __device__ int climbs_d = 0;
static __device__ int best_d = INT_MAX;
#if QUEUE
static __device__ int restart_d = 0;
#endif
// Buffer space, used for cache and maximum propagation


#if DEBUG

#if MICRO
static __device__ unsigned long long int d_lDuration = 0;
static __device__ unsigned long long int d_cDuration = 0;
static __device__ unsigned long long int d_pDuration = 0;

static __device__ long long int load_duration[128] = {0};
static __device__ long long int compute_duration[128] = {0};
static __device__ long long int propagate_duration[128] = {0};

#endif

#if MACRO
static __device__ unsigned long long int d_uDuration = 0;
static __device__ unsigned long long int d_sDuration = 0;

static __device__ long long int update_duration[128] = {0};
static __device__ long long int single_duration[128] = {0};
#endif

#endif

// Instrumentation

#define LOG( X ) { if( DEBUG ) {X();} }

// Load
static __device__ void inline load_start() {
#if MICRO
	if(threadIdx.x == 0) {load_duration[blockIdx.x] -= clock64();}
#endif
}
static __device__ void inline load_end() {
#if MICRO
	if(threadIdx.x == 0) {load_duration[blockIdx.x] += clock64();}
#endif
}

// Compute
static __device__ void inline compute_start() {
#if MICRO
	if(threadIdx.x == 0) {compute_duration[blockIdx.x] -= clock64();}
#endif
}
static __device__ void inline compute_end() {
#if MICRO
	if(threadIdx.x == 0) {compute_duration[blockIdx.x] += clock64();}
#endif
}


// Compute
static __device__ void inline propagate_start() {
#if MICRO
	if(threadIdx.x == 0) {propagate_duration[blockIdx.x] -= clock64();}
#endif
}
static __device__ void inline propagate_end() {
#if sMICRO
	if(threadIdx.x == 0) {propagate_duration[blockIdx.x] += clock64();}
#endif
}

// Single_iter
static __device__ void inline single_start() {
#if MACRO
	if(threadIdx.x == 0 && DEBUG) {single_duration[blockIdx.x] -= clock64();}
#endif
}
static __device__ void inline single_end() {
#if MACRO
	if(threadIdx.x == 0 && DEBUG) {single_duration[blockIdx.x] += clock64();}
#endif
}

// Update
static __device__ void inline update_start() {
#if MACRO
	if(threadIdx.x == 0) {update_duration[blockIdx.x] -= clock64();}
#endif
}
static __device__ void inline update_end() {
#if MACRO
	if(threadIdx.x == 0) {update_duration[blockIdx.x] += clock64();}
#endif
}


enum ThreadBufferStatus {MORE_THREADS_THAN_BUFFER,EQUAL_SIZE,MORE_BUFFER_THAN_THREADS};

// Data structure used to hold position along path
struct __align__(8) Data {
	float x,y;
};




#if QUEUE
//
// Returns a unique integer value with the initial value being 0
//
//	Synchronizes so not safe for branches
//
// @return  - Returns the next unique int
//
static __device__ inline int
nextSlice(int* __restrict__ w_buffer) {
	if(threadIdx.x==0) {
		w_buffer[0] = atomicAdd(&restart_d, SliceSize);
	}__syncthreads();
	return w_buffer[0];
}

#endif

// Allocates and initializes my global memory and shared memory.
//
//	@pos	- An array that need to be initialized and will hold our path points
//	@weight	- An array that need to be initialized and will hold our edge weights
//	@cities	- The amount of points in our graph
//
//	@return	- Returns true if initialization was successful, false otherwise.
//
template <int TileSize>
static inline __device__ bool
initMemory(const Data* &pos_d, Data* &pos, int * &weight, const int &cities) {
	// Shared memory is required to share the allocated memory
	__shared__ Data *d;
	__shared__ int *w;

	if(threadIdx.x == 0 ) {
		d = new Data[cities + 1];
		if( d != NULL ) {
			w = new int[cities];
			if( w == NULL ) {
				delete[] d;
				d = NULL;
			}
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
	LOG( propagate_start );
	int upper = min(blockDim.x,min(TileSize,cities));

	// We have to condense things down
	if(Status == MORE_THREADS_THAN_BUFFER) {

		// Compute your index and then try to shove what you have in the buffer
		const int Index = threadIdx.x % TileSize;
		w_buffer[Index] = t_val;
		__syncthreads();

		// Now try to win (someone will win)
		for(int i = 0 ; i <= (blockDim.x	/TileSize); ++i ) {
			if(t_val < w_buffer[Index]) {
				w_buffer[Index] = t_val;
			}
		}

	}else {	// Otherwise we have more than enough room!
		w_buffer[threadIdx.x] = t_val;
	}__syncthreads();

	#pragma unroll 4
	for( int i = 512; i > 32 ; i /= 2 ) {
		if (TileSize > i && blockDim.x > i) {
			int offset = (upper + 1) / 2;
			if( threadIdx.x < offset) {
				int tmp = w_buffer[threadIdx.x + offset];
				if(tmp < t_val) {
					w_buffer[threadIdx.x] = t_val = tmp;
				}
			}__syncthreads();
			upper = offset;
		}
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

		for( int i = 16; i > 0; i = i / 2 ) {
			if(threadIdx.x < i) {
				int tmp = w_buffer[threadIdx.x + i];
				if(tmp < t_val) {
					w_buffer[threadIdx.x] = t_val = tmp;
				}
			}
		}
	}__syncthreads();
	LOG( propagate_end );
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
//
//	TODO: Is it better to reverse the weight or just recompute it?
//
static inline __device__ void
reverse(int start, int end, Data* &pos, int* &weight) {

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
singleIter(Data* &pos, int* &weight, int &minchange, int &mini, int &minj, const int &cities, float* __restrict__ x_buffer, float* __restrict__ y_buffer, int* __restrict__ w_buffer) {
	LOG( single_start );

	//
	//	The tour is divided into segments.  Each segment has a length of
	//	the number of threads, except possibly the last one.
	//
	//	We traverse through the segments.  When we are in a segment each
	//	city in the segment of the tour is given to a thread.  Then we
	//	begin scanning each city from the end of the tour until we reach
	//	the current city.  Later threads will terminate this process earlier
	//	than earlier threads.
	//
	//	During each scan we will evaluate if it is better to reverse the path
	//	between the two cities.  If so we check to see if that is better than
	//	any other possible reversal we have seen.
	//
	//	After we have done this for all segments then we call update.  Update
	//	make some modification to the tour given the set of best reversals
	//	seen by each thread.
	//
	//

	for (int leading = 0; leading < cities - 2; leading += blockDim.x) {
		int i = leading + threadIdx.x;
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

		for (int trailing = cities - 1; trailing >= leading + 2; trailing -= TileSize) {

			int bound = trailing - TileSize + 1;	// The lower bound on what we can load

			//
			//	Load the shared memory cache
			//
			//	Each thread will try to load adjacent elements
			//
			LOG( load_start );
			for(int k = threadIdx.x; k < TileSize; k += blockDim.x) {
				int cache_idx = k + bound;
				if (cache_idx >= (leading + 2)) {	// Never go below the lowest city
					x_buffer[k] = pos[cache_idx].x;
					y_buffer[k] = pos[cache_idx].y;
					w_buffer[k] = weight[cache_idx];
				}
			}__syncthreads();
			LOG( load_end );

			LOG( compute_start );

			// Compute the lower bound that we can see
			int lower = bound;
			if (lower < i + 2) lower = i + 2;

			// Go over loaded cache that everyone will use
			for (int current = trailing; current >= lower; current--) {
				int cache_idx = current - bound;

				float pxj0 = x_buffer[cache_idx];
				float pyj0 = y_buffer[cache_idx];
				int change = w_buffer[cache_idx]
					+ __float2int_rn(sqrtf((pxi0 - pxj0) * (pxi0 - pxj0) + (pyi0 - pyj0) * (pyi0 - pyj0)))
					+ __float2int_rn(sqrtf((pxi1 - pxj1) * (pxi1 - pxj1) + (pyi1 - pyj1) * (pyi1 - pyj1)));

				// Shift down
				pxj1 = pxj0;
				pyj1 = pyj0;

				// If better save it and where we found it
				if (minchange > change) {
					minchange = change;
					mini = i;
					minj = current;
				}
			}__syncthreads();

			LOG( compute_end );
		}

		if (i < cities - 2) {
			minchange += weight[i];
		}
	}
	LOG( single_end );
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

	LOG( update_start );

	// For all threads, find the best change
	maximum<Status,TileSize>(minchange, cities, w_buffer);

	// If we don't have one, oh well.
	if(w_buffer[0] >= 0) {
		LOG( update_end );
		return false;
	}

	// While we have an update
	while(w_buffer[0] < 0 ) {

		// If we have multiple bests, pick one
		if (minchange == w_buffer[0]) {
			w_buffer[1] = threadIdx.x;
		}__syncthreads();

		// Get what which indices to swap
		if(threadIdx.x==w_buffer[1]) {
			w_buffer[2] = mini;
			w_buffer[3] = minj;
		}__syncthreads();

		// Give them to each thread
		int mi = w_buffer[2];
		int mj = w_buffer[3];

		// If we are overlapping the best then we can't be used
		if(!(minj < (mi - 1)) && !(mini > (mj + 1))) {
			minchange = 0;
		}

		// Reverse the path between the nodes selected
		reverse(mi+1+threadIdx.x,mj-threadIdx.x,pos,weight);

		// Fix connecting edges weights for the endpoints
		weight[mi] = -dist(mi,mi+1);
		weight[mj] = -dist(mj,mj+1);
		__syncthreads();	// Wait for everyone

		// Get the next best
		maximum<Status,TileSize>(minchange, cities, w_buffer);
	}

	LOG( update_end );

	return true;
}

//
// Given a path we randomly permute it into a new new path and then initialize the weights of the path.
//
// @pos			- The current Hamiltonian path
// @weight		- The current weight of our edges along the path
// @cities		- The number of cities along the path (excluding the end point)
static __device__ inline void
permute(Data* &pos, int* &weight, const int &cities, curandState &rndstate) {

	if (threadIdx.x == 0) {  // serial permutation
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
static __device__ inline void
cleanup(Data* &pos, int* &weight, int &local_climbs, int &best_length) {
	if (threadIdx.x == 0) {
		// Save data
		atomicAdd(&climbs_d,local_climbs);
		atomicMin(&best_d, best_length);

		// Release memory
		delete pos;
		delete weight;

#if DEBUG

#if MICRO
		atomicAdd(&d_lDuration,load_duration[blockIdx.x]);
		atomicAdd(&d_cDuration,compute_duration[blockIdx.x]);
		atomicAdd(&d_pDuration,propagate_duration[blockIdx.x]);
#endif

#if MACRO
		atomicAdd(&d_uDuration,update_duration[blockIdx.x]);
		atomicAdd(&d_sDuration,single_duration[blockIdx.x]);
#endif

#endif
	}
}


//
// Perform iterative two-opt until there can be no more swaps to reduce the path length.
//
// @pos_d	- The position of each point in the graph.
// @cities	- The number of vertices in the graph
template <ThreadBufferStatus Status, int TileSize>
static __global__ __launch_bounds__(1024, 2) void
TwoOpt(const int Restarts, const Data *pos_d, const int cities) {


	Data	*pos;
	int 	*weight;
	int 	local_climbs = 0;
	int		best_length = INT_MAX;

	curandState rndstate;
	//curand_init(blockIdx.x , 0, 0, &rndstate);

	__shared__ float x_buffer[TileSize];
	__shared__ float y_buffer[TileSize];
	__shared__ int w_buffer[TileSize];

	// Initialize the memory, if cannot then output error and exit
	if( !initMemory<TileSize>(pos_d,pos,weight,cities) ) {
		if(threadIdx.x == 0) {
			printf("Memory initialization error for block %d\n", blockIdx.x);
		}
		return;
	}


#if QUEUE
	for(int slice = nextSlice(w_buffer) ; slice < Restarts; slice = nextSlice(w_buffer)) {	// get smaller blocks
		for( int r = slice ; r < slice + SliceSize && r < Restarts ; ++r ) {
#else
	for(int r = blockIdx.x; r < Restarts; r += gridDim.x) {	// even blocks
#endif

			if( local_climbs % 10 == 0 ) {
				curand_init( blockIdx.x + gridDim.x * local_climbs , 0, 0, &rndstate);
			}
			int mini,minj,minchange;

			// Give our current path we need to permute it
			permute(pos,weight,cities,rndstate);

			// Keep applying two-opt until we reach some local
			// (or global) minimum on the length
			do {
				++local_climbs;
				minchange = mini = minj = 0;
				singleIter<TileSize>(pos, weight, minchange, mini, minj, cities, x_buffer, y_buffer, w_buffer);
			} while (update<Status,TileSize>(pos, weight, minchange, mini, minj, cities, w_buffer));

			// Calculate the length of the path
			w_buffer[0] = 0;
			__syncthreads();
			int term = 0;
			for (int i = threadIdx.x; i < cities; i += blockDim.x) {
				term += dist(i, i + 1);
			}
			atomicAdd(w_buffer,term);
			__syncthreads();

			// If better then save it to my local best
			if(threadIdx.x == 0) {
				if(w_buffer[0] < best_length) {
					best_length = w_buffer[0];
				}
			}
#if QUEUE
		}
#endif
	}


	// Release all my resources, and save the best seen
	// with any other statistics
	cleanup(pos, weight, local_climbs, best_length);
}


//
// Checks to see if an error occurred with CUDA and if so prints out the message passed and the CUDA
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


int getMaxSharedMemory( int major ) {
	if(major < 3) {
		return 16384;
	}else if(major < 5) {
		return 32768;
	}else {
		return 65536;
	}
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

	std::cout << "Compute Version " << props.major << "."  << props.minor << std::endl;
	
	/* 5.x or higher */
	int numBlocks = 0;
	int Max_Shared = 65536;
	int Max_Blocks = 32;

	const int Block_Thread_Limit = 2048 / Threads;
	
	if(props.major < 3) {
		Max_Shared = 16384;
		Max_Blocks = 8;
	}else if(props.major < 5) {
		Max_Shared = 49152;
		Max_Blocks = 16;
	}
	
	const int Block_Shared_Limit = (Max_Shared / Shared_Bytes);
	numBlocks = props.multiProcessorCount * min(Max_Blocks,min(Block_Shared_Limit,Block_Thread_Limit));

	#if QUEUE
	numBlocks = max(MaxBlocks, numBlocks );
	#endif

	return numBlocks;
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


//
// Handle ThreadBufferStatus kernel selection
//
template <int TileSize>
static float
_wrapStatus(const int Restarts, const int Threads, const Data *Pos_d, const int Cities) {

	const int Shared_Bytes = (sizeof(int) + 2*sizeof(float)) * TileSize;
	const int Blocks = min(Restarts,getMaxBlocks(Shared_Bytes + 16,Threads));
	const ThreadBufferStatus Status = (Threads > TileSize) ? MORE_THREADS_THAN_BUFFER : (Threads < TileSize) ? MORE_BUFFER_THAN_THREADS : EQUAL_SIZE;
	float time;

	const int Device_Memory = (sizeof(int) + sizeof(Data)) * (Cities + 1)* 2*Blocks;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, Device_Memory);
	CudaTest("Change heap size");

	// Output runtime configuration
	std::cout	<< "Blocks = " << Blocks
				<< ", Threads  = " << Threads
				<< ", TileSize = " << TileSize
				<< ", Status = " << getName(Status)
				<< ", Shared Bytes = " << Shared_Bytes
				<< ", Device Memory = " << Device_Memory/(1024.0f*1024.0f) << "MB" << std::endl;
	#if QUEUE
				std::cout << "SliceSize = " << SliceSize << std::endl;
	#endif

	cudaEvent_t begin,end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	cudaThreadSetCacheConfig( cudaFuncCachePreferShared  );

	cudaProfilerStart();
	switch(Status) {
		case MORE_THREADS_THAN_BUFFER:
			cudaEventRecord(begin,0);
			TwoOpt<MORE_THREADS_THAN_BUFFER,TileSize><<<Blocks,Threads>>>(Restarts,Pos_d,Cities);
			CudaTest("Kernel Call");
			cudaEventRecord(end,0);
			cudaEventSynchronize(end);
			break;
		case EQUAL_SIZE:
			cudaEventRecord(begin,0);
			TwoOpt<EQUAL_SIZE,TileSize><<<Blocks,Threads>>>(Restarts,Pos_d,Cities);
			CudaTest("Kernel Call");
			cudaEventRecord(end,0);
			cudaEventSynchronize(end);
			break;
		case MORE_BUFFER_THAN_THREADS:
			cudaEventRecord(begin,0);
			TwoOpt<MORE_BUFFER_THAN_THREADS,TileSize><<<Blocks,Threads>>>(Restarts,Pos_d,Cities);
			CudaTest("Kernel Call");
			cudaEventRecord(end,0);
			cudaEventSynchronize(end);
			break;
	};
	cudaProfilerStop();
	cudaEventElapsedTime(&time,begin,end);

	cudaEventDestroy(begin);
	cudaEventDestroy(end);

	return time;
}



//
//	Choose the parameters
//
template<int p, int i>
class Recur {
	public:
	static float recur( const int Cities, const Data *Pos, const int Restarts, const int Threads , const int TileSize ) {
		if( i == TileSize ) {
			return _wrapStatus<i>( Restarts , Threads , Pos , Cities );
		}else {
			return Recur<p,i-32>::recur( Cities , Pos , Restarts , Threads , TileSize );
		}
	}
};


//
//	Default
//
template<int p>
class Recur<p,0> {
	public:
		static float recur( const int Cities, const Data *Pos, const int Restarts, const int Threads , const int TileSize ) {
			cudaDeviceProp props;
			cudaGetDeviceProperties(&props,0);
			int sharedMemBytes = getMaxSharedMemory( props.major ) / (2 * (sizeof(int) + 2 * sizeof(float)));
			if( sharedMemBytes < 1344 && sharedMemBytes >= 1024 ) {
				return _wrapStatus<1024>(Restarts,Threads,Pos,Cities);
			} else if( sharedMemBytes < 2048 && sharedMemBytes >= 1344 ) {
				return _wrapStatus<1344>(Restarts,Threads,Pos,Cities);
			}else if( sharedMemBytes >= 2048 ) {
				return _wrapStatus<2048>(Restarts,Threads,Pos,Cities);
			}else {
				std::cout << "Invalid TileSize = " << TileSize << std::endl;
				exit(-1);
			}
			return -1;
		}
};

//
//	Auto-generate templates so I don't have to.
//
//	Runs through each possible value form 0 to 1024
//
float
RunKernel(const int Cities, const Data *Pos, const int Restarts, const int Threads, const int TileSize) {
	return Recur<1024,1024>::recur( Cities , Pos , Restarts , Threads , TileSize );
}

//
//	Main entry point to program.
//
//
//	argv[0] - program name
//	argv[1] - input file
//	argv[2] - restarts
//	argv[3] - threads
//	argv[4] - shared memory
//
int
main(int argc, char *argv[]) {

	if (argc < 3 || argc > 5) {fprintf(stderr, "\narguments: input_file restart_count <threads> <tilesize> \n"); exit(-1);}

	const int Restarts = atoi(argv[2]);
	if (Restarts < 1) {fprintf(stderr, "restart_count is too small: %d\n", Restarts); exit(-1);}

	Data *pos_d;
	const int Cities = readInput(argv[1], &pos_d);	// Load data to GPU
	printf("configuration: %d cities, %d restarts, %s input\n", Cities, Restarts, argv[1]);

	// Make sure we are a multiple of 32 and less than 1024
	const int Threads = (argc >= 4) ? min(1024,next32(atoi(argv[3]))) : min(1024,next32(Cities));

	// How big is our shared memory
	const int TileSize = (argc >= 5) ? min( next32(atoi(argv[4])),2048) : Threads;

	// Run the kernel
	const float time = RunKernel(Cities,pos_d,Restarts,Threads,TileSize);

	// Synchronize (just in case)
	cudaDeviceSynchronize();

	// how long it took
	int hours = (int)(time / (3600.0f * 1000.0f));
	int seconds = (int)(time/1000) % 60;
	int minutes = (int)((time/1000) / 60) % 60;

	// Grab the data
	int climbs,best;
	cudaMemcpyFromSymbol(&climbs,climbs_d,sizeof(int),0,cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&best,best_d,sizeof(int),0,cudaMemcpyDeviceToHost);


#if DEBUG

#if MICRO
	long long pd,cd,ld;
	cudaMemcpyFromSymbol(&pd,propagate_duration,sizeof(int),0,cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&cd,compute_duration,sizeof(int),0,cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&ld,load_duration,sizeof(int),0,cudaMemcpyDeviceToHost);
#else
	long long sd,ud;

	cudaMemcpyFromSymbol(&sd,single_duration,sizeof(int),0,cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&ud,update_duration,sizeof(int),0,cudaMemcpyDeviceToHost);
#endif

#endif

	// Output
	long long moves = 1LL * climbs * (Cities - 2) * (Cities - 1) / 2;
	std::cout << "Number of two-opts " << climbs << std::endl;
	std::cout << moves * 0.000001 / time << "Gmoves/s" << std::endl;
	std::cout << "best found tour length = " << best << std::endl;
	std::cout << "Total Time : " << time / 1000.0f << "s" << std::endl;
	std::cout << "Hours = " << hours << ", Minutes = " << minutes << ", Seconds = " << seconds << ", Milliseconds = " << (int)(time) % 1000 << std::endl;

#if DEBUG
#if MICRO
	std::cout << "Propagate: " << pd << std::endl;
	std::cout << "Load: " << ld << std::endl;
	std::cout << "Compute: " << cd << std::endl;
#else
	std::cout << "Single: " << sd << std::endl;
	std::cout << "Update: " << ud << std::endl;
#endif
#endif

	// Reset and free all the data
	cudaDeviceReset();
	cudaFree(pos_d);
	return 0;
}
