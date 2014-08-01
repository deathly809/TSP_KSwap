
#define S_DATA 1

// C++
#include <iostream>

// C
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>

// CUDA
#include <cuda.h>
#include <curand_kernel.h>

// data structures
enum ThreadBufferStatus {MORE_THREADS_THAN_BUFFER,EQUAL_SIZE,MORE_BUFFER_THAN_THREADS};

// Data structure used to hold position along path
struct __align__(8) Data {
	float x,y;
};

struct S_Data {
	int w;
	float x,y;
};

/******************************************************************************/
/*** 2-opt with random restarts ***********************************************/
/******************************************************************************/

// Global stats
static __device__ __managed__ int climbs_d = 0;
static __device__ __managed__ int best_d = INT_MAX;
static __device__ int restart_d = 0;

// Buffer space
#if S_DATA
extern __shared__ S_Data buffer[];
#else
extern __shared__ char buffer[];
__shared__ float *x_buffer;
__shared__ float *y_buffer;
__shared__ int   *w_buffer;
#endif

// Wrappers for shared memory buffer(s)
static __device__ inline void sX(const int &index, const float &v) {
#if S_DATA
	buffer[index].x = v;
#else
	x_buffer[index] = v;
#endif
}
static __device__ inline void sY(const int &index, const float &v) {
#if S_DATA
	buffer[index].y = v;
#else
	y_buffer[index] = v ;
#endif
}
static __device__ inline void sW(const int &index, const float &v) {
#if S_DATA
	buffer[index].w = v;
#else
	w_buffer[index] = v;
#endif
}
static __device__ inline float gX(const int &index) {
#if S_DATA
	return buffer[index].x;
#else
	return x_buffer[index];
#endif
}
static __device__ inline float gY(const int &index) {
#if S_DATA
	return buffer[index].y;
#else
	return y_buffer[index];
#endif
}
static __device__ inline int   gW(const int &index) {
#if S_DATA
	return buffer[index].w;
#else
	return w_buffer[index];
#endif
}

//
// Give two points returns the distance between them
//
// @x1	- X value of the first point
// @x1	- Y value of the first point
// @x2	- X value of the second point
// @y2	- Y value of the second point
//
// @return - Returns the distance between the two points given
static __device__ inline float 
dist(float &x1, float &y1, float &x2, float &y2) {
	float x = x1-x2;
	float y = y1-y2; y *= y;
	return __float2int_rn(sqrtf(x*x + y));
}

//
// Returns a unique integer value with the intial value being 0
//
// @return  - Returns the next unique integer
static __device__ int 
nextInt() {
	if(threadIdx.x==0) {
		sW(0,atomicAdd(&restart_d,1));
	}__syncthreads();
	return gW(0);
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
#if S_DATA == 0
	// Initialize shared memory
	x_buffer = (float*)buffer;
	y_buffer = (float*)(buffer + sizeof(float) * TileSize);
	w_buffer = (int*)(buffer + 2 * sizeof(float) * TileSize);
#endif

	return true;
}

//
// Each thread gives some integer value, then the "best" of them is returned.
//
// @t_val  - The number that the thread submits as a candidate for the maximum value
// @cities - The number of cities.
//
// @return - The best value of t_val seen from all threads
template <int Reductions,ThreadBufferStatus Status, int TileSize>
static inline __device__ int
maximum(int t_val, const int &cities) {
	int upper = min(cities,Status==MORE_THREADS_THAN_BUFFER?TileSize:blockDim.x);

	if(Status == MORE_THREADS_THAN_BUFFER) {
		const int Index = threadIdx.x % TileSize;
		sW(Index,t_val);
		__syncthreads();
		for(int i = -1 ; i <= blockDim.x/TileSize; ++i ) {
			sW(Index,t_val = min(t_val,gW(Index)));
		}__syncthreads();
	}else {
		sW(threadIdx.x,t_val);
		__syncthreads();
	}

	if(Reductions==10) {	// 1024
		const int offset = (upper + 1)/2;
		if(threadIdx.x<offset) {
			sW(threadIdx.x,t_val = min(t_val,gW(threadIdx.x+offset)));
		}__syncthreads();
		upper = offset;
	}
	if(Reductions>=9) {		// 512
		const int offset = (upper + 1)/2;
		if(threadIdx.x<offset) {
			sW(threadIdx.x,t_val = min(t_val,gW(threadIdx.x+offset)));
		}__syncthreads();
		upper = offset;
	}
	if(Reductions>=8) {		// 256
		const int offset = (upper + 1)/2;
		if(threadIdx.x<offset) {
			sW(threadIdx.x,t_val = min(t_val,gW(threadIdx.x+offset)));
		}__syncthreads();
		upper = offset;
	}
	if(Reductions>=7) {		// 128
		const int offset = (upper + 1)/2;
		if(threadIdx.x<offset) {
			sW(threadIdx.x,t_val = min(t_val,gW(threadIdx.x+offset)));
		}__syncthreads();
		upper = offset;
	}
	if(Reductions>=6) {		// 64
		const int offset = (upper + 1)/2;
		if(threadIdx.x<offset) {
			sW(threadIdx.x,t_val = min(t_val,gW(threadIdx.x+offset)));
		}
	}

	if(threadIdx.x < 16) {
		sW(threadIdx.x,t_val=min(t_val,gW(threadIdx.x+16)));
		sW(threadIdx.x,t_val=min(t_val,gW(threadIdx.x+ 8)));
		sW(threadIdx.x,t_val=min(t_val,gW(threadIdx.x+ 4)));
		sW(threadIdx.x,t_val=min(t_val,gW(threadIdx.x+ 2)));
		sW(threadIdx.x,t_val=min(t_val,gW(threadIdx.x+ 1)));
	}__syncthreads();


	return gW(0);
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
template <ThreadBufferStatus Status,int TileSize>
static __device__ void
singleIter(Data* &pos, int* &weight, int &minchange, int &mini, int &minj, const int &cities) {


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

			if(Status==MORE_BUFFER_THAN_THREADS) {
				for(int k = threadIdx.x; k < TileSize; k += blockDim.x) {
					int index = k + bound;
					if (index >= (ii + 2)) {
						sX(k,pos[index].x);
						sY(k,pos[index].y);
						sW(k,weight[index]);
					}
				}
			}else {
				if(threadIdx.x < TileSize) {
					int index = threadIdx.x + bound;
					if (index >= (ii + 2)) {
						sX(threadIdx.x,pos[index].x);
						sY(threadIdx.x,pos[index].y);
						sW(threadIdx.x,weight[index]);
					}
				}
			}__syncthreads();

			int lower = bound;
			if (lower < i + 2) lower = i + 2;

			for (int j = jj; j >= lower; j--) {
				int jm = j - bound;

				float pxj0 = gX(jm);
				float pyj0 = gY(jm);
				int change = gW(jm) +
					+ dist(pxi0,pyi0,pxj0,pyj0)
					+ dist(pxi1,pyi1,pxj1,pyj1);

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
template <int Reductions, ThreadBufferStatus Status, int TileSize>
static __device__ bool
update(Data* &pos, int* &weight, int &minchange, int &mini, int &minj, const int &cities) {

	maximum<Reductions,Status,TileSize>(minchange, cities);
	if(gW(0) >= 0) return false;

	if (minchange == gW(0)) {
		sW(1,threadIdx.x);
	}__syncthreads();

	if(gW(1) == threadIdx.x) {
		sW(2,mini);
		sW(3,minj);
	}__syncthreads();

	mini = gW(2);
	minj = gW(3);

	// Fix path and weights
	reverse(mini+1+threadIdx.x,minj-threadIdx.x,pos,weight);

	// Fix connecting points
	weight[mini] = -dist(pos[mini].x,pos[mini].y,pos[mini+1].x,pos[mini+1].y);
	weight[minj] = -dist(pos[minj].x,pos[minj].y,pos[minj+1].x,pos[minj+1].y);
	__syncthreads();
	return true;
}

//
// Given a path we randomly permute it into a new new path and then initialize the weights of the path.
//
// @pos			- The current Hamiltonian path
// @weight		- The current weight of our edges along the path
// @cities		- The number of cities along the path (excluding the end point)
static __device__ void permute(Data* &pos, int* &weight, const int &cities) {
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

	for (int i = threadIdx.x; i < cities; i += blockDim.x) weight[i] = -dist(pos[i].x, pos[i].y, pos[i+1].x, pos[i+1].y);
	__syncthreads();

}


//
// Perform iterative two-opt until there can be no more swaps to reduce the path length.
//
// @pos_d	- The position of each point in the graph.
// @cities	- The number of vertices in the graph
template <int Reductions,ThreadBufferStatus Status, int TileSize>
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


	for(int r = nextInt() ; r < Restarts; r = nextInt()) {

		int mini,minj,minchange;

		permute(pos,weight,cities);

		do {
			++local_climbs;
			minchange = mini = minj = 0;
			singleIter<Status,TileSize>(pos, weight, minchange, mini, minj, cities);
		} while (update<Reductions,Status,TileSize>(pos, weight, minchange, mini, minj, cities));

		__shared__ int w; w = 0;
		__syncthreads();
		int term = 0;
		for (int i = threadIdx.x; i < cities; i += blockDim.x) {
			term += dist(pos[i].x, pos[i].y, pos[i+1].x, pos[i+1].y);
		}
		atomicAdd(&w,term);
		__syncthreads();

		if(threadIdx.x==0) {
			if(w < best_length) {
				best_length = w;
			}
		}

	}

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

// Terrible (TODO: Turn into functions)
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
static const char*
getName(const ThreadBufferStatus status) {
	switch(status) {
		case MORE_THREADS_THAN_BUFFER:
			return "MORE_THREADS_THAN_BUFFER";
		case EQUAL_SIZE:
			return "EQUAL_SIZE";
		case MORE_BUFFER_THAN_THREADS:
			return "MORE_BUFFER_THAN_THREADS";
	};
	return "error";
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
// How many reductions do we need to perform in order to make sure we have found our min/max/etc
//
// @return returns the number of reductions needed to propogate any value
static int
computeReductions(const int Cities, const int Threads, const int TileSize) {
	int MaxData = min(Threads,min(TileSize,Cities));
	if(MaxData>512) return 10;
	if(MaxData>256) return 9;
	if(MaxData>128) return 8;
	if(MaxData>64) return 7;
	if(MaxData>32) return 6;
	return 5;
}

//
// Calculates the maximum number of resident blocks that the card can hold
//
// @Threads - Number of threads that each block will have
//
// @return - Returns the number of blocks the card can have resident
static int
getMaxBlocks(const int Threads) {
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,0);

	if(props.major < 3) {
		return props.multiProcessorCount * min(8,(int)(2048/Threads));
	}else if(props.major < 5) {
		return props.multiProcessorCount * min(16,(int)(2048/Threads));
	}else {
		return props.multiProcessorCount * min(32,(int)(2048/Threads));
	}
}

template <int Reductions,int TileSize>
static float
Kernel(const ThreadBufferStatus Status, const int Restarts, const int Blocks, const int Threads, const Data *Pos_d, const int Cities,const int Shared_Bytes) {
	std::cout << "Blocks = " << Blocks << ", Threads  = " << Threads << ", TileSize = " << TileSize << ", Status = " << getName(Status) << ", Reductions = " << Reductions << ", Shared Bytes = " << Shared_Bytes << std::endl;


	float time;
	cudaEvent_t begin,end;
	cudaEventCreate(&begin);
	cudaEventCreate(&end);



	switch(Status) {
		case MORE_THREADS_THAN_BUFFER:
			cudaEventRecord(begin,0);
			TwoOpt<Reductions,MORE_THREADS_THAN_BUFFER,TileSize><<<Blocks,Threads,Shared_Bytes>>>(Restarts,Pos_d,Cities);
			cudaEventRecord(end,0);
			break;
		case EQUAL_SIZE:
			cudaEventRecord(begin,0);
			TwoOpt<Reductions,EQUAL_SIZE,TileSize><<<Blocks,Threads,Shared_Bytes>>>(Restarts,Pos_d,Cities);
			cudaEventRecord(end,0);
			break;
		case MORE_BUFFER_THAN_THREADS:
			cudaEventRecord(begin,0);
			TwoOpt<Reductions,MORE_BUFFER_THAN_THREADS,TileSize><<<Blocks,Threads,Shared_Bytes>>>(Restarts,Pos_d,Cities);
			cudaEventRecord(end,0);
			break;
	};

	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time,begin,end);

	cudaEventDestroy(begin);
	cudaEventDestroy(end);

	return time;
}

//
//	Run the kernel
//
template <int TileSize>
static float
RunKernel(const int Restarts, const int Threads, const Data *Pos_d, const int Cities) {

	const int Shared_Bytes = sizeof(S_Data) * TileSize;
	const int Blocks = min(Restarts,getMaxBlocks(Threads));
	const ThreadBufferStatus Status = (Threads > TileSize) ? MORE_THREADS_THAN_BUFFER : (Threads < TileSize) ? MORE_BUFFER_THAN_THREADS : EQUAL_SIZE;
	const int Reductions = computeReductions(Cities,Threads,TileSize);

	switch(Reductions) {
		case 10:
			return Kernel<10,TileSize>(Status, Restarts, Blocks, Threads, Pos_d, Cities, Shared_Bytes);
		case 9:
			return Kernel<9,TileSize>(Status, Restarts, Blocks, Threads, Pos_d, Cities, Shared_Bytes);
		case 8:
			return Kernel<8,TileSize>(Status, Restarts, Blocks, Threads, Pos_d, Cities, Shared_Bytes);
		case 7:
			return Kernel<7,TileSize>(Status, Restarts, Blocks, Threads, Pos_d, Cities, Shared_Bytes);
		case 6:
			return Kernel<6,TileSize>(Status, Restarts, Blocks, Threads, Pos_d, Cities, Shared_Bytes);
		default:
			return Kernel<5,TileSize>(Status, Restarts, Blocks, Threads, Pos_d, Cities, Shared_Bytes);
	}
}

//
//	Main entry point to program.
//
int
main(int argc, char *argv[]) {
	printf("2-opt TSP CUDA GPU code v2.1 [Kepler]\n");
	printf("Copyright (c) 2014, Texas State University. All rights reserved.\n");

	if (argc < 3 || argc > 5) {fprintf(stderr, "\narguments: input_file restart_count <threads> <tilesize> \n"); exit(-1);}

	const int Restarts = atoi(argv[2]);
	if (Restarts < 1) {fprintf(stderr, "restart_count is too small: %d\n", Restarts); exit(-1);}

	Data *pos_d;
	const int Cities = readInput(argv[1], &pos_d);
	printf("configuration: %d cities, %d restarts, %s input\n", Cities, Restarts, argv[1]);

	const int Threads = min(1024,(argc >= 4) ? next32(atoi(argv[3])) : next32(Cities));
	const int TileSize = min(1024,(argc >= 5) ? next32(atoi(argv[4])) : Threads);

	float time;

	switch(TileSize) {
		case 32:
			time = RunKernel<32>(Restarts,Threads,pos_d,Cities);
			break;
		case 64:
			time = RunKernel<64>(Restarts,Threads,pos_d,Cities);
			break;
		case 96:
			time = RunKernel<96>(Restarts,Threads,pos_d,Cities);
			break;
		case 128:
			time = RunKernel<128>(Restarts,Threads,pos_d,Cities);
			break;
		case 160:
			time = RunKernel<160>(Restarts,Threads,pos_d,Cities);
			break;
		case 192:
			time = RunKernel<192>(Restarts,Threads,pos_d,Cities);
			break;
		case 224:
			time = RunKernel<224>(Restarts,Threads,pos_d,Cities);
			break;
		case 256:
			time = RunKernel<256>(Restarts,Threads,pos_d,Cities);
			break;
		case 288:
			time = RunKernel<288>(Restarts,Threads,pos_d,Cities);
			break;
		case 320:
			time = RunKernel<320>(Restarts,Threads,pos_d,Cities);
			break;
		case 352:
			time = RunKernel<352>(Restarts,Threads,pos_d,Cities);
			break;
		case 384:
			time = RunKernel<384>(Restarts,Threads,pos_d,Cities);
			break;
		case 416:
			time = RunKernel<416>(Restarts,Threads,pos_d,Cities);
			break;
		case 448:
			time = RunKernel<448>(Restarts,Threads,pos_d,Cities);
			break;
		case 480:
			time = RunKernel<480>(Restarts,Threads,pos_d,Cities);
			break;
		case 512:
			time = RunKernel<512>(Restarts,Threads,pos_d,Cities);
			break;
		case 544:
			time = RunKernel<544>(Restarts,Threads,pos_d,Cities);
			break;
		case 576:
			time = RunKernel<576>(Restarts,Threads,pos_d,Cities);
			break;
		case 608:
			time = RunKernel<608>(Restarts,Threads,pos_d,Cities);
			break;
		case 640:
			time = RunKernel<640>(Restarts,Threads,pos_d,Cities);
			break;
		case 672:
			time = RunKernel<672>(Restarts,Threads,pos_d,Cities);
			break;
		case 704:
			time = RunKernel<704>(Restarts,Threads,pos_d,Cities);
			break;
		case 736:
			time = RunKernel<736>(Restarts,Threads,pos_d,Cities);
			break;
		case 768:
			time = RunKernel<768>(Restarts,Threads,pos_d,Cities);
			break;
		case 800:
			time = RunKernel<800>(Restarts,Threads,pos_d,Cities);
			break;
		case 832:
			time = RunKernel<832>(Restarts,Threads,pos_d,Cities);
			break;
		case 864:
			time = RunKernel<864>(Restarts,Threads,pos_d,Cities);
			break;
		case 896:
			time = RunKernel<896>(Restarts,Threads,pos_d,Cities);
			break;
		case 928:
			time = RunKernel<928>(Restarts,Threads,pos_d,Cities);
			break;
		case 960:
			time = RunKernel<960>(Restarts,Threads,pos_d,Cities);
			break;
		case 992:
			time = RunKernel<992>(Restarts,Threads,pos_d,Cities);
			break;
		case 1024:
			time = RunKernel<1024>(Restarts,Threads,pos_d,Cities);
			break;
		default:
			std::cout << "Error : " << TileSize << std::endl;
	};
	CudaTest("kernel launch failed");  // needed for timing

	int hours = time / (3600.0f * 1000.0f);
	int seconds = (int)(time/1000) % 60;
	int minutes = (int)(time/1000) / 60;


	long long moves = 1LL * climbs_d * (Cities - 2) * (Cities - 1) / 2;

	std::cout << moves * 0.000001 / time << "Gmoves/s" << std::endl;
	std::cout << "best found tour length = " << best_d << std::endl;
	std::cout << "Total Time : " << time / 1000.0f << "s" << std::endl;
	std::cout << "Hours = " << hours << ", Minutes = " << minutes << ", Seconds = " << seconds << ", Milliseconds = " << (int)(time) % 1000 << std::endl;


	cudaDeviceReset();
	cudaFree(pos_d);
	return 0;
}

