#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <random>

#define BLOCK_SIZE 500

using namespace std;

__global__ void piCalcGPU(float* d_X, float* d_Y, int* d_countInBlocks, int blocksPerGrid, int N) 
{

  __shared__ int shared_blocks[500];

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * blocksPerGrid;

  int points_in_circle = 0;  
  for (int i = index; i < N; i+= stride) {    
	if (d_X[i]*d_X[i] + d_Y[i]*d_Y[i] <= 1.0f) {
	  points_in_circle++;    
	}  
  }
  shared_blocks[threadIdx.x] = points_in_circle;
  __syncthreads();

  if (threadIdx.x == 0) 
  {    
    int pointsInCircleBlock = 0;    
    for (int j = 0; j < blockDim.x; j++) 
    {      
      pointsInCircleBlock += shared_blocks[j];    
    }
    d_countInBlocks[blockIdx.x] = pointsInCircleBlock;  
  }
}

float piCalcCPU(int interval, float * X, float * Y) {
	int points_in_circle = 0;
  float dist;
	for(int i = 0; i < interval; i++) {
	  dist = X[i]*X[i] + Y[i]*Y[i];
    if (dist <= 1.0){
        points_in_circle++;
    }
	}
	return 4.0f * points_in_circle / interval;
}


float * generateSequencesRandom(int N) {
    float * randArr = new float[N];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int n = 0; n < N; ++n) {
        randArr[n] = dis(gen);
    }
    return randArr;
}

int main(int argc, char *argv[]) {  
  srand(time(NULL));
  
  int N = atoi(argv[1]);
  
  float * h_X = generateSequencesRandom(N);   
  float * h_Y = generateSequencesRandom(N);   
   
  size_t size = N * sizeof(float);    
  float* d_X;    
  float* d_Y;    
  cudaMalloc((void **)&d_X, size);  
  cudaMalloc((void **)&d_Y, size);
  cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice);    
  cudaMemcpy(d_Y, h_Y, size, cudaMemcpyHostToDevice);
  
  int threadsPerBlock = BLOCK_SIZE;
  int blocks = N / BLOCK_SIZE;
  int blocksPerGrid = (N % BLOCK_SIZE > 0) ?  blocks + 1 :  blocks;
  size_t countBlocks = blocksPerGrid * sizeof(int);
 
  int* d_countInBlocks;
  cudaMalloc((void **)&d_countInBlocks, countBlocks);
 
  clock_t start1 = clock();
  piCalcGPU<<<blocksPerGrid, threadsPerBlock>>>(d_X, d_Y, d_countInBlocks, blocksPerGrid, N);
  if (cudaSuccess != cudaGetLastError())
    cout << "Error!\n";

  int* h_countInBlocks = new int[blocksPerGrid];
  cudaMemcpy(h_countInBlocks, d_countInBlocks, countBlocks, cudaMemcpyDeviceToHost);

  int N_in_circle = 0;
  for (int i = 0 ; i < blocksPerGrid; i++) {
    N_in_circle = N_in_circle + h_countInBlocks[i];
  }
  float pi_gpu = 4.0 * float(N_in_circle) / N;

  clock_t stop1 = clock();
  float gpu_time = (stop1-start1)/(float)CLOCKS_PER_SEC;
  printf("time Pi GPU: %f s.\n", gpu_time);
  printf("value Pi GPU: %f\n", pi_gpu);

  clock_t start2 = clock();
  float pi_cpu = piCalcCPU(N, h_X, h_Y);
	clock_t stop2 = clock();
  float cpu_time = (stop2-start2)/(float)CLOCKS_PER_SEC;
	printf("time Pi CPU: %f s.\n", cpu_time);
  printf("value Pi CPU: %f\n", pi_cpu);

  printf("Acceleration: %f\n", cpu_time/gpu_time);

}