
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
#define dist(a, b) __float2int_rn(sqrtf((x[a] - x[b]) * (x[a] - x[b]) + (y[a] - y[b]) * (y[a] - y[b])))

static __device__ int climbs_d = 0;
static __device__ int best_d = INT_MAX;


enum ThreadBufferStatus {MORE_THREADS_THAN_BUFFER,EQUAL_SIZE,MORE_BUFFER_THAN_THREADS};

static inline __device__ void
swap( float *arr, int i, int j) {
    float t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
}

// Allocates and initializes my global memory and shared memory.
//
//	@x	    - An array that need to be initialized and will hold our x values
//	@y	    - An array that need to be initialized and will hold our y values
//	@weight	- An array that need to be initialized and will hold our edge weights
//	@cities	- The amount of points in our graph
//
//	@return	- Returns true if initialization was successful, false otherwise.
template <int TileSize>
static inline __device__ bool 
initMemory(const float* &x_d, const float* &y_d, float* &x, float* &y, int * &weight, const int &cities) {
    __shared__ float* p_x;
    __shared__ float* p_y;
    __shared__ int* w;
    // Allocate my global memory
    if(threadIdx.x == 0 ) {
        p_x = new float[cities + 1];
        if(p_x != NULL) {
            p_y = new float[cities + 1];
            if( p_y != NULL ) {
                w = new int[cities];
                if(w == NULL) {
                    delete[] p_x;
                    delete[] p_y;
                    p_x = NULL;
                    p_y = NULL;
                }
            }else{
                delete[] p_x;

            }
        }
    }__syncthreads();

    if(p_x == NULL) {
        return false;
    }

    // Save new memory locations
    x = p_x;
    y = p_y;
    weight = w;

    for (int i = threadIdx.x; i < cities; i += blockDim.x) {
        x[i] = x_d[i];
        y[i] = y_d[i];
    }
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
        if(TileSize > 32) {
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
//	@start 	    - The first position in the sub-path we have to swap with the end
// 	@end	    - The last position in the path we have to swap with the start
//	@x          - The x positions in our path
//	@y          - The y positions in our path
//	@weights    - The edge weights between points
static inline __device__ void 
reverse(int start, int end, float* &x, float * &y, int* &weight) {
    while(start<end) {

        swap( x , start , end );
        swap( y , start , end );

        int   w = weight[start];
        weight[start] = weight[end-1];
        weight[end-1] = w;

        start += blockDim.x;
        end -= blockDim.x;

    }__syncthreads();
}

//
// Perform a single iteration of Two-Opt
// @x			- The current Hamiltonian paths x coordinates
// @y			- The current Hamiltonian paths y coordinates
// @weight		- The current weight of our edges along the path
// @minchange 	- The current best change we can make
// @mini		- The ith city in the path that is part of the swap
// @minj		- The jth city in the path that is part of the swap
// @cities		- The number of cities along the path (excluding the end point)
template <int TileSize>
static __device__ void 
singleIter(float * &x, float * &y, int* &weight, int &minchange, int &mini, int &minj, const int &cities, float* __restrict__ x_buffer, float* __restrict__ y_buffer, int* __restrict__ w_buffer) {


    for (int ii = 0; ii < cities - 2; ii += blockDim.x) {
        int i = ii + threadIdx.x;
        float pxi0, pyi0, pxi1, pyi1, pxj1, pyj1;

        if (i < cities - 2) {
            minchange -= weight[i];
            pxi0 = x[i];
            pxi1 = x[i+1];

            pyi0 = y[i];
            pyi1 = y[i+1];

            pxj1 = x[0];
            pyj1 = y[0];
        }

        for (int jj = cities - 1; jj >= ii + 2; jj -= TileSize) {

            int bound = jj - TileSize + 1;

            for(int k = threadIdx.x; k < TileSize; k += blockDim.x) {
                int index = k + bound;
                if (index >= (ii + 2)) {
                    x_buffer[k] = x[index];
                    y_buffer[k] = y[index];
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
// @x			- The current Hamiltonian paths x coordinates
// @y			- The current Hamiltonian paths y coordinates
// @weight		- The current weight of our edges along the path
// @minchange 	- The current best change we can make
// @mini		- The ith city in the path that is part of the swap
// @minj		- The jth city in the path that is part of the swap
// @cities		- The number of cities along the path (excluding the end point)
template <ThreadBufferStatus Status, int TileSize>
static __device__ bool 
update(float * &x, float * &y, int* &weight, int &minchange, int &mini, int &minj, const int &cities, int* __restrict__ w_buffer) {

    maximum<Status,TileSize>(minchange, cities,w_buffer);
    if(w_buffer[0] >= 0) return false;

    if (minchange == w_buffer[0]) {
        w_buffer[1] = ((mini) << 16) + minj;  // non-deterministic winner
    }__syncthreads();

    mini = w_buffer[1] >> 16;
    minj = w_buffer[1] & 0xffff;

    // Fix path and weights
    reverse(mini+1+threadIdx.x,minj-threadIdx.x, x, y, weight);

    // Fix connecting points
    weight[mini] = -dist(mini,mini+1);
    weight[minj] = -dist(minj,minj+1);
    __syncthreads();
    return true;
}

//
// Given a path we randomly permute it into a new new path and then initialize the weights of the path.
//
// @x			- The current Hamiltonian paths x coordinates
// @y			- The current Hamiltonian paths y coordinates
// @weight		- The current weight of our edges along the path
// @cities		- The number of cities along the path (excluding the end point)
static __device__ inline void 
permute(float* &x, float * &y, int* &weight, const int &cities) {
    if (threadIdx.x == 0) {  // serial permutation
        curandState rndstate;
        curand_init(blockIdx.x, 0, 0, &rndstate);
        for (int i = 1; i < cities; i++) {
            int j = curand(&rndstate) % (cities - 1) + 1;
            swap( x , i , j );
            swap( y , i , j );
        }
        x[cities] = x[0];
        y[cities] = y[0];
    }__syncthreads();

    for (int i = threadIdx.x; i < cities; i += blockDim.x) weight[i] = -dist(i, i + 1);
    __syncthreads();

}


//
// Perform iterative two-opt until there can be no more swaps to reduce the path length.
//
// @x_d	    - The x coordinates each point in the graph.
// @y_d	    - The y coordinates each point in the graph.
// @cities	- The number of vertices in the graph
template <ThreadBufferStatus Status, int TileSize>
static __global__ __launch_bounds__(1024, 2) void 
TwoOpt(const float *x_d, const float * y_d, const int cities) {

    float	*x;
    float	*y;
    int 	*weight;
    int 	local_climbs = 0;
    int mini,minj,minchange;

    if( !initMemory<TileSize>(x_d,y_d,x,y,weight,cities) ) {
        if(threadIdx.x == 0) {
            printf("Memory initialization error for block %d\n", blockIdx.x);
        }
        return;
    }

    __shared__ float x_buffer[TileSize];
    __shared__ float y_buffer[TileSize];
    __shared__ int w_buffer[TileSize];

    permute(x,y,weight,cities);

    do {
        ++local_climbs;
        minchange = mini = minj = 0;
        singleIter<TileSize>(x,y, weight, minchange, mini, minj, cities, x_buffer, y_buffer, w_buffer);
    } while (update<Status,TileSize>(x,y, weight, minchange, mini, minj, cities, w_buffer));


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
        delete x;
        delete y;
        delete weight;
    }
}


//
// Checks to see if an error occured with CUDA and if so prints out the message passed and the CUDA 
// error then quits the application.
//
// @msg	- Message to print out if error occurs
static void 
CudaTest(std::string msg) {
    cudaError_t e;
    cudaThreadSynchronize();
    if (cudaSuccess != (e = cudaGetLastError())) {
        std::cerr << msg << ": " <<  e << std::endl;;
        std::cerr << cudaGetErrorString(e) << std::endl;;
        exit(-1);
    }
}

#define mallocOnGPU(addr, size) if (cudaSuccess != cudaMalloc((void **)&addr, size)) fprintf(stderr, "could not allocate GPU memory\n");  CudaTest("couldn't allocate GPU memory");
#define copyToGPU(to, from, size) if (cudaSuccess != cudaMemcpy(to, from, size, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of data to device failed\n");  CudaTest("data copy to device failed");

//
// Read TPS lib files into GPU memory.  ATT and CEIL_2D edge weight types are not supported
//
// @fname	- The name of the file to read the TSP data from
// @x_d	    - Pointer to the pointer that will hold x coordinate 
//            data on GPU and is modified here to be the address
//            on the GPU
//
// @return	- Returns the number of cities found
static int 
readInput(const char *fname, float **x_d, float **y_d) {
    int ch, cnt, in1, cities;
    float in2, in3;
    FILE *f;
    float *x;
    float *y;
    char str[256];  // potential for buffer overrun

    f = fopen(fname, "rt");
    if (f == NULL) {fprintf(stderr, "could not open file %s\n", fname);  exit(-1);}

    ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
    ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
    ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);

    ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
    if(fscanf(f, "%s\n", str) == 0) exit(-1);;
    cities = atoi(str);
    if (cities <= 2) {fprintf(stderr, "only %d cities\n", cities);  exit(-1);}

    x = new float[cities];  if (x == NULL) {fprintf(stderr, "cannot allocate x");  exit(-1);}
    y = new float[cities];  if (y == NULL) {fprintf(stderr, "cannot allocate y");  exit(-1);}

    ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
    if( fscanf(f, "%s\n", str) == 0 ) exit( -1 );;
    if (strcmp(str, "NODE_COORD_SECTION") != 0) {fprintf(stderr, "wrong file format\n");  exit(-1);}

    cnt = 0;
    while (fscanf(f, "%d %f %f\n", &in1, &in2, &in3)) {

        x[cnt] = in2;
        y[cnt] = in3;

        ++cnt;

        if (cnt > cities) {fprintf(stderr, "input too long\n");  exit(-1);}
        if (cnt != in1) {fprintf(stderr, "input line mismatch: expected %d instead of %d\n", cnt, in1);  exit(-1);}
    }
    if (cnt != cities) {fprintf(stderr, "read %d instead of %d cities\n", cnt, cities);  exit(-1);}

    if( fscanf(f, "%s", str) == 0 ) exit( -1 );
    if (strcmp(str, "EOF") != 0) {fprintf(stderr, "didn't see 'EOF' at end of file\n");  exit(-1);}

    // X
    mallocOnGPU(*x_d, sizeof(float) * cities);
    copyToGPU(*x_d, x, sizeof(float) * cities);

    // Y
    mallocOnGPU(*y_d, sizeof(float) * cities);
    copyToGPU(*y_d, y, sizeof(float) * cities);

    fclose(f);

    delete[] x;
    delete[] y;

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
//	Run the kernel
//
template <int TileSize>
static float 
RunKernel(const int Restarts, const int Threads, const float *x_d, const float *y_d, const int Cities) {
    float time;
    cudaEvent_t begin,end;
    const int Shared_Bytes = (sizeof(int) + 2*sizeof(float)) * TileSize;
    const int Blocks = Restarts;
    const ThreadBufferStatus Status = (Threads > TileSize) ? MORE_THREADS_THAN_BUFFER : (Threads < TileSize) ? MORE_BUFFER_THAN_THREADS : EQUAL_SIZE;

    // Weight and position
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,(sizeof(int) + 2 * sizeof(float)) * (Cities +1 ) * Blocks);
    CudaTest("cudaDeviceSetLimit");

    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    CudaTest( "Creating events" );

    std::cout << "Blocks = " << Blocks << ", Threads  = " << Threads << ", TileSize = " << TileSize << ", Status = " << getName(Status) << ", Shared Bytes = " << Shared_Bytes << std::endl;

    cudaEventRecord(begin,0);
    switch(Status) {
        case MORE_THREADS_THAN_BUFFER:
            TwoOpt<MORE_THREADS_THAN_BUFFER,TileSize><<<Restarts,Threads>>>(x_d,y_d,Cities);
            break;
        case EQUAL_SIZE:
            TwoOpt<EQUAL_SIZE,TileSize><<<Restarts,Threads>>>(x_d,y_d,Cities);
            break;
        case MORE_BUFFER_THAN_THREADS:
            TwoOpt<MORE_BUFFER_THAN_THREADS,TileSize><<<Restarts,Threads>>>(x_d,y_d,Cities);
            break;
    };
    cudaEventRecord(end,0);

    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time,begin,end);

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    
    CudaTest( "Finished Kernel" );

    return time;
}

//
// Given an integer returns the next multiple of 32 greater than or equal to it.
//
// @in 		- The integer to round to next multiple of 32
//
// @return 	- Returns the next multiple of 32 that is greater than or equals to in
static int 
next32(int in) {
    return ((in + 31) >> 5) << 5;
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

    float *x_d;
    float *y_d;
    const int Cities = readInput(argv[1], &x_d, &y_d);
    printf("configuration: %d cities, %d restarts, %s input\n", Cities, Restarts, argv[1]);



    if (Cities > MAXCITIES) {
        fprintf(stderr, "the problem size is too large for this version of the code\n");
    } else {

        const int Threads = (argc >= 4) ? min(1024,next32(atoi(argv[3]))) : min(1024,next32(Cities));
        const int TileSize = (argc >= 5) ? min( next32(atoi(argv[4])),1024) : Threads;

        float time;

        switch(TileSize) {
            case 32:
                time = RunKernel<32>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 64:
                time = RunKernel<64>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 96:
                time = RunKernel<96>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 128:
                time = RunKernel<128>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 160:
                time = RunKernel<160>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 192:
                time = RunKernel<192>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 224:
                time = RunKernel<224>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 256:
                time = RunKernel<256>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 288:
                time = RunKernel<288>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 320:
                time = RunKernel<320>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 352:
                time = RunKernel<352>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 384:
                time = RunKernel<384>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 416:
                time = RunKernel<416>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 448:
                time = RunKernel<448>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 480:
                time = RunKernel<480>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 512:
                time = RunKernel<512>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 544:
                time = RunKernel<544>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 576:
                time = RunKernel<576>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 608:
                time = RunKernel<608>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 640:
                time = RunKernel<640>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 672:
                time = RunKernel<672>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 704:
                time = RunKernel<704>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 736:
                time = RunKernel<736>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 768:
                time = RunKernel<768>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 800:
                time = RunKernel<800>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 832:
                time = RunKernel<832>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 864:
                time = RunKernel<864>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 896:
                time = RunKernel<896>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 928:
                time = RunKernel<928>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 960:
                time = RunKernel<960>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 992:
                time = RunKernel<992>(Restarts,Threads,x_d,y_d,Cities);
                break;
            case 1024:
                time = RunKernel<1024>(Restarts,Threads,x_d,y_d,Cities);
                break;
            default:
                std::cout << "Error : " << TileSize << std::endl;
        };
        CudaTest("kernel launch failed");  // needed for timing

        int hours = (int)(time / (3600.0f * 1000.0f));
        int seconds = (int)(time/1000) % 60;
        int minutes = (int)(time/1000) / 60;

        int climbs;
        int best;

        cudaMemcpyFromSymbol( &climbs   , climbs_d  , sizeof(int) );
        cudaMemcpyFromSymbol( &best     , best_d    , sizeof(int) );

        long long moves = 1LL * climbs * (Cities - 2) * (Cities - 1) / 2;

        std::cout << moves * 0.000001 / time << "Gmoves/s" << std::endl;
        std::cout << "best found tour length = " << best << std::endl;
        std::cout << "Total Time : " << time / 1000.0f << "s" << std::endl;
        std::cout << "Hours = " << hours << ", Minutes = " << minutes << ", Seconds = " << seconds << " Milliseconds = " << (int)(time) % 1000 << std::endl;
    }

    cudaDeviceReset();
    cudaFree(x_d);
    cudaFree(y_d);
    return 0;
}

