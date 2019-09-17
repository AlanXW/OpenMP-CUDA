#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include "device_functions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "act17xw"		//replace with your user name

void print_help();
int process_command_line(int argc, char *argv[]);

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;
char file_input[256];
char file_output[256];
char *optional = "PPM_BINARY";
int c = 0;
MODE execution_mode = CPU;

struct rgb {
	unsigned char r;
	unsigned char g;
	unsigned char b;
};
typedef struct rgb rgb;

struct mosaic {
	int r;
	int g;
	int b;
};
typedef struct mosaic mosaic;

__device__ int total_colour[3];

// 1.0  Basic function
// 1 cell = 1 block, 1 pixel = 1 thread, c can not over 32 because the number of threads is up to 1024
__global__ void mosaicAver_1(rgb *output, mosaic *m_output, int *dc) {
	int threadId = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	int blockId = blockIdx.y*gridDim.x + blockIdx.x;
	
	atomicAdd(&m_output[blockId].r, output[threadId].r);
	atomicAdd(&m_output[blockId].g, output[threadId].g);
	atomicAdd(&m_output[blockId].b, output[threadId].b);
	
	atomicAdd(&dc[0], output[threadId].r);
	atomicAdd(&dc[1], output[threadId].g);
	atomicAdd(&dc[2], output[threadId].b);
			
	output[threadId].r = m_output[blockId].r / blockDim.x / blockDim.y;
	output[threadId].g = m_output[blockId].g / blockDim.x / blockDim.y;
	output[threadId].b = m_output[blockId].b / blockDim.x / blockDim.y;
}

// 2.0	Reducing data replication
__global__ void mosaicAver_2(rgb *output, mosaic *m_output) {
	int threadId = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	int blockId = blockIdx.y*gridDim.x + blockIdx.x;

	atomicAdd(&m_output[blockId].r, output[threadId].r);
	atomicAdd(&m_output[blockId].g, output[threadId].g);
	atomicAdd(&m_output[blockId].b, output[threadId].b);

	atomicAdd(&total_colour[0], output[threadId].r);
	atomicAdd(&total_colour[1], output[threadId].g);
	atomicAdd(&total_colour[2], output[threadId].b);

	output[threadId].r = m_output[blockId].r / blockDim.x / blockDim.y;
	output[threadId].g = m_output[blockId].g / blockDim.x / blockDim.y;
	output[threadId].b = m_output[blockId].b / blockDim.x / blockDim.y;
}

// 3.0  Remove unnecessary summation
__global__ void mosaicAver_3(rgb *output, mosaic *m_output) {
	int threadId = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	int blockId = blockIdx.y*gridDim.x + blockIdx.x;

	atomicAdd(&m_output[blockId].r, output[threadId].r);
	atomicAdd(&m_output[blockId].g, output[threadId].g);
	atomicAdd(&m_output[blockId].b, output[threadId].b);
	// Only one thread per block run in this step. 
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		atomicAdd(&total_colour[0], m_output[blockId].r);
		atomicAdd(&total_colour[1], m_output[blockId].g);
		atomicAdd(&total_colour[2], m_output[blockId].b);
	}

	output[threadId].r = m_output[blockId].r / blockDim.x / blockDim.y;
	output[threadId].g = m_output[blockId].g / blockDim.x / blockDim.y;
	output[threadId].b = m_output[blockId].b / blockDim.x / blockDim.y;
}

// 4.0  Shared Memory
__global__ void mosaicAver_4(rgb *output, mosaic *m_output) {

	// Assign m_output to shared memory
	__shared__ mosaic m_s;

	int threadId = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	int blockId = blockIdx.y*gridDim.x + blockIdx.x;
	
	atomicAdd(&m_output[blockId].r, output[threadId].r);
	atomicAdd(&m_output[blockId].g, output[threadId].g);
	atomicAdd(&m_output[blockId].b, output[threadId].b);

	m_s.r = m_output[blockId].r;
	m_s.g = m_output[blockId].g;
	m_s.b = m_output[blockId].b;
	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		atomicAdd(&total_colour[0], m_s.r);
		atomicAdd(&total_colour[1], m_s.g);
		atomicAdd(&total_colour[2], m_s.b);
	}

	output[threadId].r = m_s.r / blockDim.x / blockDim.y;
	output[threadId].g = m_s.g / blockDim.x / blockDim.y;
	output[threadId].b = m_s.b / blockDim.x / blockDim.y;
}

// 5.0  Advanced Shared Memory
__global__ void mosaicAver_5(rgb *output, mosaic *m_output) {

	__shared__ mosaic m_s;
	// assign output to shared memory
	extern __shared__ rgb output_s[];

	int threadId = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
	int blockId = blockIdx.y*gridDim.x + blockIdx.x;
	int threadId_1d = threadIdx.y*blockDim.x + threadIdx.x;

	output_s[threadId_1d] = output[threadId];
	__syncthreads();

	atomicAdd(&m_output[blockId].r, output_s[threadId_1d].r);
	atomicAdd(&m_output[blockId].g, output_s[threadId_1d].g);
	atomicAdd(&m_output[blockId].b, output_s[threadId_1d].b);

	m_s.r = m_output[blockId].r;
	m_s.g = m_output[blockId].g;
	m_s.b = m_output[blockId].b;
	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		atomicAdd(&total_colour[0], m_s.r);
		atomicAdd(&total_colour[1], m_s.g);
		atomicAdd(&total_colour[2], m_s.b);
	}

	output[threadId].r = m_s.r / blockDim.x / blockDim.y;
	output[threadId].g = m_s.g / blockDim.x / blockDim.y;
	output[threadId].b = m_s.b / blockDim.x / blockDim.y;
}

int main(int argc, char *argv[]) {
	FILE *f_input;
	FILE *f_output;
	double begin, end;
	double seconds;
	char height_buff[5];
	char width_buff[5];
	char read_type[3];
	static int height = 0;
	static int width = 0;
	int t_height;
	int t_width;
	int arg_header = 0;
	int rsum;
	int gsum;
	int bsum;
	int raver;
	int gaver;
	int baver;
	int rsum_total = 0;
	int gsum_total = 0;
	int bsum_total = 0;
	int read_buff;
	char header[1024];
	rgb *rgb_output;
	rgb *d_rgb_output;
	mosaic *rgb_mosaic;

		if (process_command_line(argc, argv) == FAILURE)
			return 1;

	//TODO: read input image file (either binary or plain text PPM) 
	f_input = fopen(file_input, "rb");
	if (f_input == NULL) {
		fprintf(stderr, "Error: Could not find %s file \n", file_input);
		exit(1);
	}
	if (c > 8) c = 8;

	// Read header of the file and skip comments
	// Use fscanf to read header because we don't know whether the file divide width and height by space or '\n'
	while (arg_header < 4) {
		fscanf(f_input, "%s", header);
		if (header[0] != '#') {
			if (arg_header == 0) {
				sscanf(header, "%s", read_type);
			}
			if (arg_header == 1) {
				sscanf(header, "%s", width_buff);
			}
			if (arg_header == 2) {
				sscanf(header, "%s", height_buff);
			}
			arg_header++;
		}
		else {
			fgets(header, 50, f_input);
		}
	}
	width = atoi(width_buff);
	height = atoi(height_buff);
	rgb_output = (rgb *)malloc(sizeof(rgb)*width*height);

	// Computing the whole picture is divided into many small tasks by c
	t_height = height / c;
	t_width = width / c;
	// Skip '\n' at the end of header because fscanf doesn't read '\n'
	fgets(header, 20, f_input);
	// Read pixel by two ways
	if (strcmp(read_type, "P3") == 0) {
		for (int i = 0; i < width*height; i++) {
			fscanf(f_input, "%d", &read_buff);
			rgb_output[i].r = (char)read_buff;
			fscanf(f_input, "%d", &read_buff);
			rgb_output[i].g = (char)read_buff;
			fscanf(f_input, "%d", &read_buff);
			rgb_output[i].b = (char)read_buff;
		}
	}
	else if (strcmp(read_type, "P6") == 0)
		fread(rgb_output, sizeof(char), 3 * width*height, f_input);
	fclose(f_input);



	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode) {
	case (CPU): {
		//TODO: starting timing here
		begin = omp_get_wtime();
		//TODO: calculate the average colour value
		for (int i = 0; i<t_height; i++)
			for (int j = 0; j<t_width; j++) {
				// In each small task, we need to calculate the average RGB value of the pixels in every small region and assign the value to each pixel in the region. The size of the region depends on c.
				rsum = 0;
				gsum = 0;
				bsum = 0;
				// Locate the first pixel in each region. According to the index of the first pixel, the index of each pixel to be calculated can be obtained.
				int index_first = (j + i * width) * c;
				for (int h = 0; h < c; h++)
					for (int w = 0; w < c; w++) {
						rsum += rgb_output[index_first + (h*width) + w].r;
						gsum += rgb_output[index_first + (h*width) + w].g;
						bsum += rgb_output[index_first + (h*width) + w].b;
					}
				// Add up the total RBG values of all regions
				rsum_total += rsum;
				gsum_total += gsum;
				bsum_total += bsum;
				raver = rsum / (c*c);
				gaver = gsum / (c*c);
				baver = bsum / (c*c);
				for (int h = 0; h<c; h++)
					for (int w = 0; w < c; w++) {
						rgb_output[index_first + (h*width) + w].r = raver;
						rgb_output[index_first + (h*width) + w].g = gaver;
						rgb_output[index_first + (h*width) + w].b = baver;
					}
			}
		// Output the average colour value for the image
		printf("CPU Average image colour red = %d, green = %d, blue = %d \n", rsum_total / (width*height), gsum_total / (width*height), bsum_total / (width*height));

		//TODO: end timing here
		end = omp_get_wtime();
		seconds = end - begin;
		printf("CPU mode execution time took %d s and %fms\n", (int)seconds, (seconds - (int)seconds) * 1000);
		break;
	}
	case (OPENMP): {
		int index_first;
		int i, j, h, w;
		//TODO: starting timing here
		begin = omp_get_wtime();
		//TODO: calculate the average colour value	
		// Computing logic is the same as CPU mode. On this basis, OPENMP parallel computing is added.
#pragma omp parallel for private(j,index_first,rsum,gsum,bsum,raver,gaver,baver,h,w)
		for (i = 0; i<t_height; i++)
			for (j = 0; j<t_width; j++) {
				rsum = 0;
				gsum = 0;
				bsum = 0;
				index_first = (j + i * width) * c;

				for (h = 0; h < c; h++)
					for (w = 0; w < c; w++) {
						rsum += rgb_output[index_first + (h*width) + w].r;
						gsum += rgb_output[index_first + (h*width) + w].g;
						bsum += rgb_output[index_first + (h*width) + w].b;
					}
				raver = rsum / (c*c);
				gaver = gsum / (c*c);
				baver = bsum / (c*c);

				for (h = 0; h<c; h++)
					for (w = 0; w < c; w++) {
						rgb_output[index_first + (h*width) + w].r = raver;
						rgb_output[index_first + (h*width) + w].g = gaver;
						rgb_output[index_first + (h*width) + w].b = baver;
					}
#pragma omp critical
				{
					rsum_total += rsum;
					gsum_total += gsum;
					bsum_total += bsum;
				}

			}
		// Output the average colour value for the image
		printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n", rsum_total / (width*height), gsum_total / (width*height), bsum_total / (width*height));

		//TODO: end timing here
		end = omp_get_wtime();
		seconds = end - begin;
		printf("OPENMP mode execution time took %d s and %fms\n", (int)seconds, (seconds - (int)seconds) * 1000);
		break;
	}
	case (CUDA): {
		cudaEvent_t start, stop;
		float milliseconds = 0;
		int bc[3];
		// Start timing
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		// Copy data from CPU to GPU and allocate memory
		cudaMalloc((void **)&d_rgb_output, sizeof(rgb)*width*height);
		cudaMalloc((void **)&rgb_mosaic, sizeof(mosaic)*width / c * height / c);
		cudaMemcpy(d_rgb_output, rgb_output, sizeof(rgb)*width*height, cudaMemcpyHostToDevice);

		// 1 block = 1 mosaic cell, 1 thread = 1 pixel
		dim3 blocksPerGrid(width/c, height/c, 1);
		dim3 threadsPerBlock(c, c, 1);

		cudaEventRecord(start);
		// Kernel function
		mosaicAver_4 << < blocksPerGrid, threadsPerBlock >> > (d_rgb_output, rgb_mosaic);
		cudaDeviceSynchronize();

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);

		cudaMemcpy(rgb_output, d_rgb_output, sizeof(rgb)*width*height, cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(bc, total_colour, sizeof(int) * 3);
		// End timing
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		// Release
		cudaFree(d_rgb_output);
		cudaFree(rgb_mosaic);

		printf("CUDA  Average image colour red = %d, green = %d, blue = %d \n", bc[0]/(width*height) , bc[1]/ (width*height), bc[2]/ (width*height));

		printf("CUDA mode execution time took %fms\n", milliseconds);
		break;
	}
	case (ALL): {
		int index_first;
		int i, j, h, w;
		// starting timing, this block is for CPU Mode
		// same as CPU Mode
		begin = omp_get_wtime();
		for (i = 0; i<t_height; i++)
			for (j = 0; j<t_width; j++) {
				rsum = 0;
				gsum = 0;
				bsum = 0;
				// Locate the first pixel in each region. According to the index of the first pixel, the index of each pixel to be calculated can be obtained.
				index_first = (j + i * width) * c;
				for (h = 0; h < c; h++)
					for (w = 0; w < c; w++) {
						rsum += rgb_output[index_first + (h*width) + w].r;
						gsum += rgb_output[index_first + (h*width) + w].g;
						bsum += rgb_output[index_first + (h*width) + w].b;
					}
				// Add up the total RBG values of all regions
				rsum_total += rsum;
				gsum_total += gsum;
				bsum_total += bsum;
				raver = rsum / (c*c);
				gaver = gsum / (c*c);
				baver = bsum / (c*c);
				for (h = 0; h<c; h++)
					for (w = 0; w < c; w++) {
						rgb_output[index_first + (h*width) + w].r = raver;
						rgb_output[index_first + (h*width) + w].g = gaver;
						rgb_output[index_first + (h*width) + w].b = baver;
					}
			}
		// Output the average colour value for the image
		printf("CPU Average image colour red = %d, green = %d, blue = %d \n", rsum_total / (width*height), gsum_total / (width*height), bsum_total / (width*height));
		// end timing
		end = omp_get_wtime();
		seconds = end - begin;
		printf("CPU mode execution time took %d s and %fms\n", (int)seconds, (seconds - (int)seconds) * 1000);
		//starting timing, this block is for OpenMP Mode
		begin = omp_get_wtime();
		//same as OpenMP Mode
		rsum_total = 0;
		gsum_total = 0;
		bsum_total = 0;
#pragma omp parallel for private(j,index_first,rsum,gsum,bsum,raver,gaver,baver,h,w)
		for (i = 0; i<t_height; i++)
			for (j = 0; j<t_width; j++) {
				rsum = 0;
				gsum = 0;
				bsum = 0;
				index_first = (j + i * width) * c;

				for (h = 0; h < c; h++)
					for (w = 0; w < c; w++) {
						rsum += rgb_output[index_first + (h*width) + w].r;
						gsum += rgb_output[index_first + (h*width) + w].g;
						bsum += rgb_output[index_first + (h*width) + w].b;
					}
				raver = rsum / (c*c);
				gaver = gsum / (c*c);
				baver = bsum / (c*c);

				for (h = 0; h<c; h++)
					for (w = 0; w < c; w++) {
						rgb_output[index_first + (h*width) + w].r = raver;
						rgb_output[index_first + (h*width) + w].g = gaver;
						rgb_output[index_first + (h*width) + w].b = baver;
					}
#pragma omp critical
				{
					rsum_total += rsum;
					gsum_total += gsum;
					bsum_total += bsum;
				}

			}
		// Output the average colour value for the image
		printf("OPENMP Average image colour red = %d, green = %d, blue = %d \n", rsum_total / (width*height), gsum_total / (width*height), bsum_total / (width*height));

		// end timing here
		end = omp_get_wtime();
		seconds = end - begin;
		printf("OPENMP mode execution time took %d s and %fms\n", (int)seconds, (seconds - (int)seconds) * 1000);

		// CUDA Mode
		cudaEvent_t start, stop;
		float milliseconds = 0;
		int bc[3];

		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaMalloc((void **)&d_rgb_output, sizeof(rgb)*width*height);
		cudaMalloc((void **)&rgb_mosaic, sizeof(mosaic)*width / c * height / c);
		cudaMemcpy(d_rgb_output, rgb_output, sizeof(rgb)*width*height, cudaMemcpyHostToDevice);

		dim3 blocksPerGrid(width / c, height / c, 1);
		dim3 threadsPerBlock(c, c, 1);

		cudaEventRecord(start);
		mosaicAver_4 << < blocksPerGrid, threadsPerBlock >> > (d_rgb_output, rgb_mosaic);
		cudaDeviceSynchronize();

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&milliseconds, start, stop);

		cudaMemcpy(rgb_output, d_rgb_output, sizeof(rgb)*width*height, cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(bc, total_colour, sizeof(int) * 3);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaFree(d_rgb_output);
		cudaFree(rgb_mosaic);

		printf("CUDA  Average image colour red = %d, green = %d, blue = %d \n", bc[0] / (width*height), bc[1] / (width*height), bc[2] / (width*height));
		//end timing
		printf("CUDA mode execution time took %fms\n", milliseconds);
		break;
	}
	}

	//save the output image file (from last executed mode)
	f_output = fopen(file_output, "wb");
	if (strcmp(optional, "PPM_BINARY") == 0) {
		fprintf(f_output, "P6\n%d\n%d\n255\n", width, height);
		fwrite(rgb_output, sizeof(char), 3 * width*height, f_output);
	}
	else if (strcmp(optional, "PPM_PLAIN_TEXT") == 0) {
		fprintf(f_output, "P3\n%d\n%d\n255\n", width, height);
		for (int i = 0; i < width*height; i++) {
			fprintf(f_output, "%d ", (int)rgb_output[i].r);
			fprintf(f_output, "%d ", (int)rgb_output[i].g);
			fprintf(f_output, "%d\t", (int)rgb_output[i].b);
			if (((i + 1) % width) == 0) {
				fprintf(f_output, "\n");
			}
		}
	}
	else {
		printf("Wrong output format!");
	}

	fclose(f_output);
	free(rgb_output);

	return 0;
}

void print_help() {
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		"\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		"\t               ALL. The mode specifies which version of the simulation\n"
		"\t               code should execute. ALL should execute each mode in\n"
		"\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		"\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		"\t               PPM_PLAIN_TEXT\n ");
}

int process_command_line(int argc, char *argv[]) {
	if (argc < 7) {
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//first argument is always the executable name

	//read in the non optional command line arguments
	c = (int)atoi(argv[1]);
	// Output warning if c is not positive power of 2 number.
	if (c % 2 != 0 || c < 0) {
		printf("C should be any positive power of 2 number.\n");
		return FAILURE;
	}
	//TODO: read in the mode
	if (strcmp(argv[2], "CPU") == 0) {
		execution_mode = CPU;
	}
	else if (strcmp(argv[2], "OPENMP") == 0) {
		execution_mode = OPENMP;
	}
	else if (strcmp(argv[2], "CUDA") == 0) {
		execution_mode = CUDA;
	}
	else if (strcmp(argv[2], "ALL") == 0) {
		execution_mode = ALL;
	}
	else {
		printf("Wrong Mode Parameter!");
	}


	//TODO: read in the input image name
	if (strcmp(argv[3], "-i") == 0)
		strcpy(file_input, argv[4]);


	//TODO: read in the output image name
	if (strcmp(argv[5], "-o") == 0)
		strcpy(file_output, argv[6]);

	//TODO: read in any optional part 3 arguments
	if (argv[7] != NULL) {
		if (strcmp(argv[7], "-f") == 0) {
			optional = argv[8];
		}
	}

	return SUCCESS;
}
