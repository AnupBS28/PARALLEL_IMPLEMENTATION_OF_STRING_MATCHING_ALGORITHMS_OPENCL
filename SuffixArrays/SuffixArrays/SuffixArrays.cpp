#include <stdio.h>
#include <CL/cl2.hpp>
#include <stdlib.h>
#include <cstdlib>
#pragma warning(disable : 4996)
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <streambuf>
#include <algorithm>
using namespace std;
#define MAX_SOURCE_SIZE 0x1000000
#define MERGESORT_SMALL_STRIDE 1024 * 64
struct suffix
{
	int index;
	int rank[2];
};
int cmp(struct suffix a, struct suffix b)
{
	return (a.rank[0] == b.rank[0]) ? (a.rank[1] < b.rank[1] ? 1 : 0) : (a.rank[0] < b.rank[0] ? 1 : 0);
}
size_t GetGlobalWorkSize(size_t DataElemCount, size_t LocalWorkSize)
{
	size_t r = DataElemCount % LocalWorkSize;
	if (r == 0)
		return DataElemCount;
	else
		return DataElemCount + LocalWorkSize - r;
}
void Sort_Mergesort(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], struct suffix *m_hInput, cl_kernel m_MergesortStartKernel, cl_kernel m_MergesortGlobalSmallKernel, cl_kernel m_MergesortGlobalBigKernel, cl_mem m_dPingArray, cl_mem m_dPongArray, size_t m_N_padded)
{

	cl_int clError;
	size_t globalWorkSize[1];
	size_t localWorkSize[1];

	localWorkSize[0] = LocalWorkSize[0];
	globalWorkSize[0] = GetGlobalWorkSize(m_N_padded / 2, localWorkSize[0]);
	unsigned int locLimit = 1;
	clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N_padded * sizeof(struct suffix), m_hInput, 0, NULL, NULL);
	if (m_N_padded >= LocalWorkSize[0] * 2)
	{
		locLimit = 2 * LocalWorkSize[0];

		clError = clSetKernelArg(m_MergesortStartKernel, 0, sizeof(cl_mem), (void *)&m_dPingArray);
		clError |= clSetKernelArg(m_MergesortStartKernel, 1, sizeof(cl_mem), (void *)&m_dPongArray);

		clError = clEnqueueNDRangeKernel(CommandQueue, m_MergesortStartKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

		swap(m_dPingArray, m_dPongArray);
	}

	unsigned int stride = 2 * locLimit;

	localWorkSize[0] = LocalWorkSize[0];
	globalWorkSize[0] = GetGlobalWorkSize(m_N_padded / 2, localWorkSize[0]);

	if (m_N_padded <= MERGESORT_SMALL_STRIDE)
	{

		clError = clSetKernelArg(m_MergesortGlobalSmallKernel, 3, sizeof(cl_uint), (void *)&m_N_padded);

		for (; stride <= m_N_padded; stride <<= 1)
		{

			size_t neededWorkers = m_N_padded / stride;

			localWorkSize[0] = min(LocalWorkSize[0], neededWorkers);
			globalWorkSize[0] = GetGlobalWorkSize(neededWorkers, localWorkSize[0]);

			clError = clSetKernelArg(m_MergesortGlobalSmallKernel, 0, sizeof(cl_mem), (void *)&m_dPingArray);
			clError |= clSetKernelArg(m_MergesortGlobalSmallKernel, 1, sizeof(cl_mem), (void *)&m_dPongArray);
			clError |= clSetKernelArg(m_MergesortGlobalSmallKernel, 2, sizeof(cl_uint), (void *)&stride);

			clError = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalSmallKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

			swap(m_dPingArray, m_dPongArray);

			/*for (int yu = 0;yu < m_N_padded;yu++)
			{
				printf("%d %d %d m_hInput \n", m_hInput[yu].rank[0], m_hInput[yu].rank[1], m_hInput[yu].index);
			}*/
		}
	}
	else
	{

		clError = clSetKernelArg(m_MergesortGlobalBigKernel, 3, sizeof(cl_uint), (void *)&m_N_padded);

		for (; stride <= m_N_padded; stride <<= 1)
		{

			size_t neededWorkers = m_N_padded / stride;

			localWorkSize[0] = min(LocalWorkSize[0], neededWorkers);
			globalWorkSize[0] = GetGlobalWorkSize(neededWorkers, localWorkSize[0]);

			clError = clSetKernelArg(m_MergesortGlobalBigKernel, 0, sizeof(cl_mem), (void *)&m_dPingArray);
			clError |= clSetKernelArg(m_MergesortGlobalBigKernel, 1, sizeof(cl_mem), (void *)&m_dPongArray);
			clError |= clSetKernelArg(m_MergesortGlobalBigKernel, 2, sizeof(cl_uint), (void *)&stride);

			clError = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalBigKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);

			swap(m_dPingArray, m_dPongArray);
		}
	}
	clEnqueueReadBuffer(CommandQueue, m_dPingArray, CL_TRUE, 0, m_N_padded * sizeof(struct suffix), m_hInput, 0, NULL, NULL);
}

int *buildSuffixArray(char *txt, int n)
{

	struct suffix *suffixes;
	suffixes = (struct suffix *)malloc(sizeof(struct suffix) * n);

	for (int i = 0; i < n; i++)
	{
		suffixes[i].index = i;
		suffixes[i].rank[0] = txt[i] - 'a';
		suffixes[i].rank[1] = ((i + 1) < n) ? (txt[i + 1] - 'a') : -1;
	}

	sort(suffixes, suffixes + n, cmp);

	int *ind;
	ind = (int *)malloc(sizeof(int) * n);

	for (int k = 4; k < 2 * n; k = k * 2)
	{

		int rank = 0;
		int prev_rank = suffixes[0].rank[0];
		suffixes[0].rank[0] = rank;
		ind[suffixes[0].index] = 0;

		for (int i = 1; i < n; i++)
		{

			if (suffixes[i].rank[0] == prev_rank &&
				suffixes[i].rank[1] == suffixes[i - 1].rank[1])
			{
				prev_rank = suffixes[i].rank[0];
				suffixes[i].rank[0] = rank;
			}
			else
			{
				prev_rank = suffixes[i].rank[0];
				suffixes[i].rank[0] = ++rank;
			}
			ind[suffixes[i].index] = i;
		}

		for (int i = 0; i < n; i++)
		{
			int nextindex = suffixes[i].index + k / 2;
			suffixes[i].rank[1] = (nextindex < n) ? suffixes[ind[nextindex]].rank[0] : -1;
		}

		sort(suffixes, suffixes + n, cmp);
	}

	int *suffixArr = new int[n];
	for (int i = 0; i < n; i++)
		suffixArr[i] = suffixes[i].index;

	return suffixArr;
}

void printArr(int arr[], int n)
{
	for (int i = 0; i < n; i++)
		cout << arr[i] << " ";
	cout << endl;
}

int main(int argc, char **argv)
{

	
	cl_kernel m_MergesortStartKernel;
	cl_kernel m_MergesortGlobalSmallKernel;
	cl_kernel m_MergesortGlobalBigKernel;

	cl_mem m_dPingArray;
	cl_mem m_dPongArray;
	size_t m_N;
	size_t m_N_padded;

	size_t LocalWorkSize[3] = {256, 1, 1};

	FILE *infile1, *infile2;
	char *a, *b;
	char *program_log;

	infile1 = fopen("input.txt", "r");

	if (infile1 == NULL)
	{
		printf("File Not Found!\n");
		return -1;
	}

	fseek(infile1, 0L, SEEK_END);
	long int n = ftell(infile1);

	fseek(infile1, 0L, SEEK_SET);
	clock_t t;
	clock_t begin;
	clock_t end;

	cout << n << endl;
	char *txt;
	txt = (char *)calloc(n, sizeof(char));

	if (txt == NULL)
		return 1;

	fread(txt, sizeof(char), n, infile1);
	fclose(infile1);

	struct suffix *suffixes;
	suffixes = (struct suffix *)malloc(sizeof(struct suffix) * n);
	begin = clock();
	int *sequential_array = buildSuffixArray(txt, n);
	end = clock();
	cout << "sequential  array output :" << n << endl;

	double ti = 0.0;
	ti = (double)(end - begin) / CLOCKS_PER_SEC;

	std::cout << "sequential time spent:" << ti << endl;

	FILE *fp;
	char *source_str;
	size_t source_size;
	fp = fopen("kernel.cl", "r");
	if (!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		getchar();
		exit(1);
	}
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;

	cl_int ret = clGetPlatformIDs(1, &platform_id, NULL);
	if (ret < 0)
	{
		printf("pm po %d", ret);
		exit(1);
	}
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
						 NULL);

	if (ret < 0)
	{
		printf("pm rt %d", ret);
		exit(1);
	}
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL,
										 &ret);
	if (ret < 0)
	{
		printf("pm bner %d", ret);
		exit(1);
	}

	cl_command_queue command_queue = clCreateCommandQueue(context,
														  device_id,
														  NULL, &ret);

	cl_mem m_txt = clCreateBuffer(context,
								  CL_MEM_READ_ONLY, n * sizeof(char), NULL, &ret);
	if (ret < 0)
	{
		printf("pm op %d", ret);
		exit(1);
	}
	cl_mem m_suffixes = clCreateBuffer(context,
									   CL_MEM_READ_WRITE, n * sizeof(struct suffix), NULL, &ret);
	if (ret < 0)
	{
		printf("pm ml %d", ret);
		exit(1);
	}
	cl_mem m_n = clCreateBuffer(context,
								CL_MEM_READ_ONLY, 1 * sizeof(int), NULL, &ret);
	if (ret < 0)
	{
		printf("pm gh %d", ret);
		exit(1);
	}
	cl_mem m_k = clCreateBuffer(context,
								CL_MEM_READ_ONLY, 1 * sizeof(int), NULL, &ret);
	if (ret < 0)
	{
		printf("pm gh %d", ret);
		exit(1);
	}
	cl_mem m_indices = clCreateBuffer(context,
									  CL_MEM_READ_ONLY, n * sizeof(int), NULL, &ret);
	if (ret < 0)
	{
		printf("pm gh %d", ret);
		exit(1);
	}
	cl_int clError, clError2;
	m_dPingArray = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(struct suffix) * n, NULL, &clError2);
	ret = clError2;
	m_dPongArray = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(struct suffix) * n, NULL, &clError2);
	ret |= clError2;
	if (ret < 0)
	{
		printf("pm ghry %d", ret);
		exit(1);
	}

	ret = clEnqueueWriteBuffer(command_queue, m_txt, CL_TRUE, 0, n * sizeof(char), txt, 0, NULL, NULL);

	ret |= clEnqueueWriteBuffer(command_queue, m_n, CL_TRUE, 0, 1 * sizeof(int), &n, 0, NULL, NULL);
	if (ret < 0)
	{
		printf("pm qe %d", ret);
		exit(1);
	}

	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
	if (ret < 0)
	{
		printf("pm n %d", ret);
		exit(1);
	}
	size_t log_size;
	ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (ret < 0)
	{

		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
							  0, NULL, &log_size);
		program_log = (char *)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
							  log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}

	cl_kernel kernel = clCreateKernel(program, "init", &ret);
	if (ret < 0)
	{
		printf("pm bn %d", ret);
		exit(1);
	}
	cl_kernel rank_to_suffix = clCreateKernel(program, "rank_to_suffix", &ret);
	if (ret < 0)
	{
		printf("pm bn2 %d", ret);
		exit(1);
	}
	cl_kernel later = clCreateKernel(program, "later", &ret);
	if (ret < 0)
	{
		printf("pm bn2 %d", ret);
		exit(1);
	}

	m_MergesortGlobalSmallKernel = clCreateKernel(program, "Sort_MergesortGlobalSmall", &clError);

	ret = clError;
	m_MergesortGlobalBigKernel = clCreateKernel(program, "Sort_MergesortGlobalBig", &clError);
	ret |= clError;

	m_MergesortStartKernel = clCreateKernel(program, "Sort_MergesortStart", &clError);
	ret |= clError;
	if (ret < 0)
	{
		printf("pm bn2 %d", ret);
		exit(1);
	}
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&m_suffixes);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&m_txt);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&m_n);
	if (ret < 0)
	{
		printf("pm qw %d", ret);
		exit(1);
	}

	if (ret < 0)
	{
		printf("pm qw %d", ret);
		exit(1);
	}

	ret = clSetKernelArg(later, 0, sizeof(cl_mem), (void *)&m_suffixes);
	ret |= clSetKernelArg(later, 1, sizeof(cl_mem), (void *)&m_indices);
	ret |= clSetKernelArg(later, 2, sizeof(cl_mem), (void *)&m_n);

	if (ret < 0)
	{
		printf("pm qw %d", ret);
		exit(1);
	}

	size_t local_item_size = 256;
	size_t global_item_size = GetGlobalWorkSize(n, local_item_size);

	begin = clock();
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

	ret = clEnqueueReadBuffer(command_queue, m_suffixes, CL_TRUE, 0, n * sizeof(struct suffix), suffixes, 0, NULL, NULL);
	Sort_Mergesort(context, command_queue, LocalWorkSize, suffixes, m_MergesortStartKernel, m_MergesortGlobalSmallKernel, m_MergesortGlobalBigKernel, m_dPingArray, m_dPongArray, n);

	int *ind;
	ind = (int *)malloc(sizeof(int) * n);

	unsigned int hj = n;
	ret = clSetKernelArg(rank_to_suffix, 2, sizeof(cl_uint), (void *)&hj);
	for (int k = 4; k < 2 * n; k = k * 2)
	{

		int rank = 0;
		int prev_rank = suffixes[0].rank[0];
		suffixes[0].rank[0] = rank;
		ind[suffixes[0].index] = 0;
		unsigned int y = k / 2;
		ret |= clSetKernelArg(rank_to_suffix, 3, sizeof(cl_uint), (void *)&y);
		for (int i = 1; i < n; i++)
		{

			if (suffixes[i].rank[0] == prev_rank &&
				suffixes[i].rank[1] == suffixes[i - 1].rank[1])
			{
				prev_rank = suffixes[i].rank[0];
				suffixes[i].rank[0] = rank;
			}
			else
			{
				prev_rank = suffixes[i].rank[0];
				suffixes[i].rank[0] = ++rank;
			}
			ind[suffixes[i].index] = i;
		}

		ret |= clEnqueueWriteBuffer(command_queue, m_indices, CL_TRUE, 0, n * sizeof(int), ind, 0, NULL, NULL);

		ret |= clEnqueueWriteBuffer(command_queue, m_suffixes, CL_TRUE, 0, n * sizeof(struct suffix), suffixes, 0, NULL, NULL);

		ret |= clSetKernelArg(rank_to_suffix, 0, sizeof(cl_mem), (void *)&m_suffixes);
		ret |= clSetKernelArg(rank_to_suffix, 1, sizeof(cl_mem), (void *)&m_indices);
		ret = clEnqueueNDRangeKernel(command_queue, rank_to_suffix, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

		ret = clEnqueueReadBuffer(command_queue, m_suffixes, CL_TRUE, 0, n * sizeof(struct suffix), suffixes, 0, NULL, NULL);

		Sort_Mergesort(context, command_queue, LocalWorkSize, suffixes, m_MergesortStartKernel, m_MergesortGlobalSmallKernel, m_MergesortGlobalBigKernel, m_dPingArray, m_dPongArray, n);
	}

	int *suffixArr = new int[n];
	for (int i = 0; i < n; i++)
		suffixArr[i] = suffixes[i].index;

	end = clock();

	int tik;
	tik = 0;
	for (int i = 0; i < n; i++)
	{
		if (suffixArr[i] != sequential_array[i])
		{
			printf("%d wro\n", i);
			tik = 1;
			break;
		}
	}
	if (tik)
		printf("wrong\n");
	else
		printf("right\n");
	double time_spent = 0.0;
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	std::cout << "parallel time spent:" << time_spent;

	clFlush(command_queue);

	ret = clReleaseKernel(kernel);
	ret = clReleaseKernel(rank_to_suffix);
	ret = clReleaseKernel(later);
	ret = clReleaseProgram(program);

	ret = clReleaseMemObject(m_indices);
	ret = clReleaseMemObject(m_k);
	ret = clReleaseMemObject(m_n);
	ret = clReleaseMemObject(m_suffixes);
	ret = clReleaseMemObject(m_txt);

	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
}