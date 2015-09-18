
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

#define MAXCITIES 32765
#define dist(a, b) __float2int_rn(sqrtf((pos[a].x - pos[b].x) * (pos[a].x - pos[b].x) + (pos[a].y - pos[b].y) * (pos[a].y - pos[b].y)))
#define swap(a, b) {float tmp = a;  a = b;  b = tmp;}

static __device__ __managed__ int climbs_d = 0;
static __device__ __managed__ int best_d = INT_MAX;
static __device__ int restart_d = 0;
// Buffer space, used for cache and maximum propagation

enum ThreadBufferStatus {MORE_THREADS_THAN_BUFFER,EQUAL_SIZE,MORE_BUFFER_THAN_THREADS};

// Data structure used to hold position along path
struct __align__(8) Data {
	float x,y;
};

//
// Returns a unique integer value with the intial value being 0
//
// @return  - Returns the next unique int
static __device__ inline int
nextInt(int* __restrict__ w_buffer) {
	if(threadIdx.x==0) {
		w_buffer[0] = atomicAdd(&restart_d,1);
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
	__shared__ int* w;
	// Allocate my global memory
	if(threadIdx.x == 0 ) {
		d = new Data[cities + 1];
		if(d != NULL) {
			w = new int[cities];
			if(w == NULL) {
				delete d;
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
maximum(int t_val, const int &cities, int* __restrict__ w_buffer) {
	
	int upper = min(blockDim.x,min(TileSize,cities));
	
	if(Status == MORE_THREADS_THAN_BUFFER) {
		int Index = threadIdx.x % TileSize;
		w_buffer[Index] = t_val;
		__syncthreads();
		for(int i = 0 ; i <= (blockDim.x/TileSize); ++i ) {
			w_buffer[Index] = t_val = min(t_val,w_buffer[Index]);
		}
	}else {
		w_buffer[threadIdx.x] = t_val;
	}__syncthreads();
	
	// 1024
	if (TileSize > 512) {
		int offset = (upper + 1) / 2;	// 200
		if( threadIdx.x < offset) {
			w_buffer[threadIdx.x] = t_val = min(t_val,w_buffer[threadIdx.x + offset]);
		}__syncthreads();
		upper = offset;
	}
	
	// 512
	if (TileSize > 256) {
		int offset = (upper + 1) / 2; // 100
		if( threadIdx.x < offset) {
			w_buffer[threadIdx.x] = t_val = min(t_val,w_buffer[threadIdx.x + offset]);
		}__syncthreads();
		upper = offset;
	}
	
	// 256
	if (TileSize > 128) {
		int offset = (upper + 1) / 2; // 50
		if( threadIdx.x < offset) {
			w_buffer[threadIdx.x] = t_val = min(t_val,w_buffer[threadIdx.x + offset]);
		}__syncthreads();
		upper = offset;
	}
	
	// 128
	if (TileSize > 64) {
		int offset = (upper + 1) / 2; // 25
		if( threadIdx.x < offset) {
			w_buffer[threadIdx.x] = t_val = min(t_val,w_buffer[threadIdx.x + offset]);
		}__syncthreads();
		upper = offset;
	}
	
	// 64 and down
	if(threadIdx.x < 32) {
		if(TileSize > 32 && blockDim.x > 32) {
			w_buffer[threadIdx.x] = t_val = min(t_val,w_buffer[threadIdx.x+(upper+1)/2]);
		}
		if(threadIdx.x < 16) {
			w_buffer[threadIdx.x] = t_val = min(t_val,w_buffer[threadIdx.x+16]);
		}
		if(threadIdx.x < 8) {
			w_buffer[threadIdx.x] = t_val = min(t_val,w_buffer[threadIdx.x+8]);
		}
		if(threadIdx.x < 4) {
			w_buffer[threadIdx.x] = t_val = min(t_val,w_buffer[threadIdx.x+4]);
		}
		if(threadIdx.x < 2) {
			w_buffer[threadIdx.x] = t_val = min(t_val,w_buffer[threadIdx.x+2]);
		}
		if(threadIdx.x < 1) {
			w_buffer[threadIdx.x] = t_val = min(t_val,w_buffer[threadIdx.x+1]);
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
			
			for(int k = threadIdx.x; k < TileSize; k += blockDim.x) {
				int index = k + bound;
				if (index >= (ii + 2)) {
					x_buffer[k] = pos[index].x;
					y_buffer[k] = pos[index].y;
					w_buffer[k] = weight[index];
				}
			}__syncthreads();

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
			}__syncthreads();
		}

		if (i < cities - 2) {
			minchange += weight[i];
		}
	}
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

	maximum<Status,TileSize>(minchange, cities, w_buffer);
	if(w_buffer[0] >= 0) return false;
	
	if (minchange == w_buffer[0]) {
		//if(atomicCAS(w_buffer,minchange,1) == minchange) {
		{
			//w_buffer[1] = mini;
			//w_buffer[2] = minj;
			w_buffer[1] = ((mini) << 16) + minj;  // non-deterministic winner
		}
	}__syncthreads();

	mini = w_buffer[1] >> 16;
	minj = w_buffer[1] & 0xFFFF;
	//mini = w_buffer[1];
	//minj = w_buffer[2];
	
	// Fix path and weights
	reverse(mini+1+threadIdx.x,minj-threadIdx.x,pos,weight);
	
	// Fix connecting points
	weight[mini] = -dist(mini,mini+1);
	weight[minj] = -dist(minj,minj+1);
	__syncthreads();
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
static __device__ inline void
cleanup(Data* &pos, int* &weight, int &local_climbs, int &best_length) {
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
static __global__ __launch_bounds__(1024, 2) void
TwoOpt(const int Restarts, const Data *pos_d, const int cities) {
	
	Data	*pos;
	int 	*weight;
	int 	local_climbs = 0;
	int		best_length = INT_MAX;
	
	if( !initMemory<TileSize>(pos_d,pos,weight,cities) ) {
		if(threadIdx.x == 0) {
			printf("Memory initialization error for block %d\n", blockIdx.x);
		}
		return;
	}
	
	__shared__ float	x_buffer[TileSize];
	__shared__ float	y_buffer[TileSize];
	__shared__ int		w_buffer[TileSize];
	
	for(int r = nextInt(w_buffer) ; r < Restarts; r = nextInt(w_buffer)) {
	
		int mini,minj,minchange;
		
		permute(pos,weight,cities);
	  
		do {
			++local_climbs;
			minchange = mini = minj = 0;
			singleIter<TileSize>(pos, weight, minchange, mini, minj, cities, x_buffer, y_buffer, w_buffer);
		} while (update<Status,TileSize>(pos, weight, minchange, mini, minj, cities, w_buffer));
	
		w_buffer[0] = 0;
		__syncthreads();
		int term = 0;
		for (int i = threadIdx.x; i < cities; i += blockDim.x) {
			term += dist(i, i + 1);
		}
		atomicAdd(w_buffer,term);
		__syncthreads();

		if(threadIdx.x==0) {
			if(w_buffer[0] < best_length) {
				best_length = w_buffer[0];
			}
		}
		
	}
	cleanup(pos, weight, local_climbs, best_length);
}


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

	int Max_Shared = 0;
	int Max_Blocks = 0;
	
	if(props.major < 3) {
		Max_Shared = 16384;
		Max_Blocks = 8;
	}else if(props.major < 5) {
		Max_Shared = 32768;
		Max_Blocks = 16;
	}else {
		Max_Shared = 65536;
		Max_Blocks = 32;
	}
	
	std::cout << "Number of processors: " << props.multiProcessorCount << std::endl;
	std::cout << "Maximum Shared Memory Size: " << Max_Shared << std::endl;
	std::cout << "Maximum Number of Blocks: " << Max_Blocks << std::endl;
	std::cout << "Requested Shared Memory: " << Shared_Bytes << std::endl;
	const int Block_Shared_Limit = (Max_Shared / Shared_Bytes);
	
	return props.multiProcessorCount * min(Max_Blocks,min(Block_Shared_Limit,(int)(2048/Threads)));
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
	const int Blocks = min(Restarts,getMaxBlocks(Shared_Bytes,Threads));
	const ThreadBufferStatus Status = (Threads > TileSize) ? MORE_THREADS_THAN_BUFFER : (Threads < TileSize) ? MORE_BUFFER_THAN_THREADS : EQUAL_SIZE;
	float time;

	const int DeviceBytes = 2 * (sizeof(int) + sizeof(Data)) * (Cities + 1) * Blocks;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, DeviceBytes);
	CudaTest("cudaDeviceSetLimit");
	
	// Output runtime configuration
	std::cout << "Blocks = " << Blocks << ", Threads  = " << Threads << ", TileSize = " << TileSize << ", Status = " << getName(Status) << ", Shared Bytes = " << Shared_Bytes << ", Device Bytes = " << DeviceBytes << std::endl;

	cudaEvent_t begin,end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);

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
	cudaEventElapsedTime(&time,begin,end);

	cudaEventDestroy(begin);
	cudaEventDestroy(end);

	return time;
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

	if (Cities > MAXCITIES) {
		fprintf(stderr, "the problem size is too large for this version of the code\n");
	} else {
  
		const int Threads = (argc >= 4) ? min(1024,next32(atoi(argv[3]))) : min(1024,next32(Cities));
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
	}

	cudaDeviceReset();
	cudaFree(pos_d);
	return 0;
}

