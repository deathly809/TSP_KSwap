
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

/******************************************************************************/
/*** 2-opt with random restarts ***********************************************/
/******************************************************************************/

#define MAXCITIES 32765
#define dist(a, b) __float2int_rn(sqrtf((pos[a].x - pos[b].x) * (pos[a].x - pos[b].x) + (pos[a].y - pos[b].y) * (pos[a].y - pos[b].y)))
#define swap(a, b) {float tmp = a;  a = b;  b = tmp;}

static __device__ __managed__ int climbs_d = 0;
static __device__ __managed__ int best_d = INT_MAX;

// Buffer space, used for cache and maximum propagation
extern __shared__ char buffer[];	// Our pool of memory to hand out to other three
__shared__ float *x_buffer;
__shared__ float *y_buffer;
__shared__ int   *w_buffer;

enum ThreadBufferStatus {MORE_THREADS_THAN_BUFFER,EQUAL_SIZE,MORE_BUFFER_THAN_THREADS};

// Data structure used to hold position along path
struct __align__(16) Data {
	float x,y;
};


// Allocates and initializes my global memory and shared memory.
//
//	@pos	- An array that need to be initialized and will hold our path points
//	@weight	- An array that need to be initialized and will hold our edge weights
//	@cities	- The amount of points in our graph
//
//	@return	- Returns true if initialization was successful, false otherwise.
template <int TileSize> static inline __device__ bool initMemory(const Data* &pos_d, Data* &pos, int * &weight, const int &cities) {
	__shared__ Data *d;
	// Allocate my global memory
	if(threadIdx.x == 0 ) {
		d = new Data[cities + 1];
		if(d != NULL) {
			w_buffer = new int[cities];
			if(w_buffer == NULL) {
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
	weight = w_buffer;
	
	for (int i = threadIdx.x; i < cities; i += blockDim.x) pos[i] = pos_d[i];
	__syncthreads();
	
	// Initialize shared memory
	x_buffer = (float*)buffer;
	y_buffer = (float*)(buffer + sizeof(float) * TileSize);
	w_buffer = (int*)(buffer + 2 * sizeof(float) * TileSize);
	
	return true;
}

//
// Each thread gives some integer value, then the maximum of them is returned.
//
// @t_val  - The number that the thread submits as a candidate for the maximum value
// @cities - The number of cities.
//
// @return - The maximum value of t_val seen from all threads
template <ThreadBufferStatus Status, int TileSize> static inline __device__ int maximum(int t_val, const int &cities) {
	
	int upper = min(blockDim.x,min(TileSize,cities));
	const int Index = (Status == MORE_THREADS_THAN_BUFFER)?(threadIdx.x%TileSize):threadIdx.x;
	w_buffer[Index] = t_val;
	__syncthreads();
	
	if(Status == MORE_THREADS_THAN_BUFFER) {
		for(int i = 0 ; i <= TileSize / blockDim.x; ++i ) {
			if(w_buffer[Index] < t_val) {
				w_buffer[Index] = t_val;
			}
		}__syncthreads();
	}
	
	if (TileSize > 512) {
		int offset = (upper + 1) / 2;
		if( threadIdx.x + offset < upper ) {
			w_buffer[Index] = t_val = min(t_val,w_buffer[Index + offset]);
		}__syncthreads();
		upper = offset;
	}
	if (TileSize > 256) {
		int offset = (upper + 1) / 2;
		if( threadIdx.x + offset < upper ) {
			w_buffer[Index] = t_val = min(t_val,w_buffer[Index + offset]);
		}__syncthreads();
		upper = offset;
	}
	if (TileSize > 128) {
		int offset = (upper + 1) / 2;
		if( threadIdx.x + offset < upper ) {
			w_buffer[Index] = t_val = min(t_val,w_buffer[Index + offset]);
		}__syncthreads();
		upper = offset;
	}
	if (TileSize > 64) {
		int offset = (upper + 1) / 2;
		if( threadIdx.x + offset < upper ) {
			w_buffer[Index] = t_val = min(t_val,w_buffer[Index + offset]);
		}__syncthreads();
		upper = offset;
	}
	if(TileSize > 32) {
		int offset = (upper + 1) / 2;
		if( threadIdx.x + offset < upper ) {
			w_buffer[Index] = t_val = min(t_val,w_buffer[Index + offset]);
		}__syncthreads();
		upper = offset;
	}
			
	int offset = (upper + 1) / 2;
	if( threadIdx.x + offset < upper ) {
		w_buffer[Index] = t_val = min(t_val,w_buffer[Index + offset]);
	}__syncthreads();
	upper = offset;
	
	offset = (upper + 1) / 2;
	if( threadIdx.x + offset < upper ) {
		w_buffer[Index] = t_val = min(t_val,w_buffer[Index + offset]);
	}__syncthreads();
	upper = offset;
	
	offset = (upper + 1) / 2;
	if( threadIdx.x + offset < upper ) {
		w_buffer[Index] = t_val = min(t_val,w_buffer[Index + offset]);
	}__syncthreads();
	upper = offset;
	
	offset = (upper + 1) / 2;
	if( threadIdx.x + offset < upper ) {
		w_buffer[Index] = t_val = min(t_val,w_buffer[Index + offset]);
	}__syncthreads();
	upper = offset;
	
	offset = (upper + 1) / 2;
	if( threadIdx.x + offset < upper ) {
		w_buffer[Index] = t_val = min(t_val,w_buffer[Index + offset]);
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
static inline __device__ void reverse(int start, int end, Data* &pos, int* &weight) {
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
static __device__ void singleIter(Data* &pos, int* &weight, int &minchange, int &mini, int &minj, const int &cities) {
	

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
template <ThreadBufferStatus Status, int TileSize> static __device__ bool update(Data* &pos, int* &weight, int &minchange, int &mini, int &minj, const int &cities) {

	maximum<Status,TileSize>(minchange, cities);
	if(w_buffer[0] >= 0) return false;
	
	if (minchange == w_buffer[0]) {
		w_buffer[1] = ((mini) << 16) + minj;  // non-deterministic winner
	}__syncthreads();

	mini = w_buffer[1] >> 16;
	minj = w_buffer[1] & 0xffff;
	
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
	
	for (int i = threadIdx.x; i < cities; i += blockDim.x) weight[i] = -dist(i, i + 1);
	__syncthreads();
	
}


//
// Perform iterative two-opt until there can be no more swaps to reduce the path length.
//
// @pos_d	- The position of each point in the graph.
// @cities	- The number of vertices in the graph
template <ThreadBufferStatus Status, int TileSize> static __global__ __launch_bounds__(1024, 2) void TwoOpt(const Data *pos_d, const int cities) {
	
	Data	*pos;
	int 	*weight;
	int 	local_climbs = 0;
	int mini,minj,minchange;
	
	if( !initMemory<TileSize>(pos_d,pos,weight,cities) ) {
		if(threadIdx.x == 0) {
			printf("Memory initialization error for block %d\n", blockIdx.x);
		}
		return;
	}
	
	permute(pos,weight,cities);
  
	do {
		++local_climbs;
    	minchange = mini = minj = 0;
		singleIter<TileSize>(pos, weight, minchange, mini, minj, cities);
	} while (update<Status,TileSize>(pos, weight, minchange, mini, minj, cities));
	
	
	w_buffer[0] = 0;
	__syncthreads();
	int term = 0;
	for (int i = threadIdx.x; i < cities; i += blockDim.x) {
		term += dist(i, i + 1);
	}
	atomicAdd(w_buffer,term);
	__syncthreads();

  if (threadIdx.x == 0) {
	// Save data
	atomicAdd(&climbs_d,local_climbs);
    atomicMin(&best_d, w_buffer[0]);
	
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
static void CudaTest(char *msg) {
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
static int readInput(const char *fname, Data **pos_d) {
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

static const char* getName(const ThreadBufferStatus status) {
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
//	Run the kernel
//
template <int TileSize>
static float RunKernel(const int Restarts, const int Threads, const Data *Pos_d, const int Cities) {
	float time;
	cudaEvent_t begin,end;
	const int Shared_Bytes = (sizeof(int) + 2 * sizeof(float)) * TileSize;
	//const int MaxBlocks = getMaxBlocks(Threads);
	const int Blocks = Restarts;
	const ThreadBufferStatus Status = (Threads > TileSize) ? MORE_THREADS_THAN_BUFFER : (Threads < TileSize) ? MORE_BUFFER_THAN_THREADS : EQUAL_SIZE;
	
	cudaEventCreate(&begin);
	cudaEventCreate(&end);
	
	std::cout << "Blocks = " << Blocks << ", Threads  = " << Threads << ", TileSize = " << TileSize << ", Status = " << getName(Status) << std::endl;
	
	cudaEventRecord(begin,0);
	switch(Status) {
		case MORE_THREADS_THAN_BUFFER:
			TwoOpt<MORE_THREADS_THAN_BUFFER,TileSize><<<Restarts,Threads,Shared_Bytes>>>(Pos_d,Cities);
			break;
		case EQUAL_SIZE:
			TwoOpt<EQUAL_SIZE,TileSize><<<Restarts,Threads,Shared_Bytes>>>(Pos_d,Cities);
			break;
		case MORE_BUFFER_THAN_THREADS:
			TwoOpt<MORE_BUFFER_THAN_THREADS,TileSize><<<Restarts,Threads,Shared_Bytes>>>(Pos_d,Cities);
			break;
	};
	cudaEventRecord(end,0);
	
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time,begin,end);
	
	cudaEventDestroy(begin);
	cudaEventDestroy(end);
	
	return time;
}

static int next32(int in) {
	return ((in + 31) >> 5) << 5;
}

//
//	Main entry point to program.
//
static int getMaxBlocks(const int Threads) {
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,0);
	return props.multiProcessorCount * min(16,2048/Threads);
}

int main(int argc, char *argv[])
{
	printf("2-opt TSP CUDA GPU code v2.1 [Kepler]\n");
	printf("Copyright (c) 2014, Texas State University. All rights reserved.\n");

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
		std::cout << "Hours = " << hours << ", Minutes = " << minutes << ", Seconds = " << seconds << " Milliseconds = " << (int)(time) % 1000 << std::endl;
	}

	cudaDeviceReset();
	cudaFree(pos_d);
	return 0;
}
