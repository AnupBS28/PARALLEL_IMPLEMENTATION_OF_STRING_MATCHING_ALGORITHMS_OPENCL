#include <iostream>
#include <string>
#include <stdio.h>
#include <CL/cl2.hpp>
#include <stdlib.h>
#include <fstream>
#include <streambuf>
#pragma warning(disable : 4996)
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define MAX_SOURCE_SIZE 0x1000000

int patternLength;

int searchFirst(char searchWord[100], int sub)
{
	int i;
	int subLength = patternLength - sub;
	for (i = 0; i < subLength; i++)
	{
		if (searchWord[i] != searchWord[sub + i])
		{
			return -1;
		}
	}
	return 0;
}

int search(char searchWord[100], int sub)
{
	int prev = searchWord[sub - 1];
	int subLength = patternLength - sub;

	for (int i = sub - 1; i >= 0; --i)
	{
		if (i - 1 >= 0)
		{
			if (prev == searchWord[i - 1])
				continue;
		}
		int flag = 0;
		int count = 0;
		for (int j = i; j <= i + subLength - 1; ++j)
		{

			if (searchWord[j] != searchWord[sub + count])
			{
				flag = 1;
			}
			count++;
		}

		if (flag == 1)
			continue;
		else
			return i;
	}
	return -1;
}

int max(int x, int y)
{
	if (x > y)
		return x;
	else
		return y;
}

int main(int argc, char **argv)
{
	int numberOfProcesses = 2;
	int i = 0, j = 0, textLength;
	int numberOfWords = 0;

	std::ifstream ifs("inputEd.txt");
	std::string str1((std::istreambuf_iterator<char>(ifs)),
					 (std::istreambuf_iterator<char>()));

	std::ifstream ifs2("input1Search.txt");
	std::string str2((std::istreambuf_iterator<char>(ifs2)),
					 (std::istreambuf_iterator<char>()));

	textLength = str1.length();
	patternLength = str2.length();

	char *A = (char *)malloc(sizeof(char) * (textLength + 1));
	strcpy(A, str1.c_str());

	std::cout << A;

	while (A[i++] != '\0')
	{
		if (A[i] == ' ')
			numberOfWords++;
	}

	int *len = (int *)malloc(sizeof(int) * (numberOfWords + 1));

	for (i = 0; i < (numberOfWords + 1); i++)
	{
		len[i] = 0;
	}

	i = 0;
	while (j < numberOfWords + 1)
	{
		while (A[i] != ' ' && A[i] != '\0')
		{
			len[j]++;
			i++;
		}
		j++;
		i++;
	}

	int numWordsPerProcess = (numberOfWords + 1) / numberOfProcesses;
	int *start_endi = (int *)malloc(sizeof(int) * (numberOfProcesses * 2));
	int count = 0;
	int startIndex;
	int endIndex;
	int k = 0;
	int lengthindex = 0;

	for (i = 0; i < numberOfProcesses; i++)
	{
		startIndex = count;
		endIndex = startIndex;
		for (j = lengthindex; j < lengthindex + numWordsPerProcess; j++)
		{

			endIndex += len[j];
		}
		lengthindex += numWordsPerProcess;
		endIndex = endIndex + numWordsPerProcess - 1;
		start_endi[k++] = startIndex;
		start_endi[k++] = endIndex - 1;
		count = endIndex + 1;
	}

	int fl = 0;
	char word[100];
	int f, sub, result;

	strcpy(word, str2.c_str());

	int dlen = str1.length();
	int *goodSymTab = (int *)malloc(sizeof(int) * (patternLength));
	int badSymTab[128];

	for (i = 0; i <= 127; i++)
	{
		badSymTab[i] = patternLength;
	}

	for (i = 0; i <= patternLength - 2; i++)
	{
		badSymTab[(int)word[i]] = patternLength - 1 - i;
	}

	for (k = 1; k <= patternLength - 1; k++)
	{
		sub = patternLength - k;
		f = 0;
		result = search(word, sub);
		if (result >= 0)
		{
			goodSymTab[k] = sub - result;
			continue;
		}
		for (sub = patternLength - k + 1; sub <= patternLength - 1; sub++)
		{
			result = searchFirst(word, sub);
			if (result == 0)
			{
				goodSymTab[k] = sub - result;
				f = 1;
				break;
			}
		}
		if (f == 0)
		{

			goodSymTab[k] = patternLength;
		}
	}

	FILE *fp;
	char *source_str;
	size_t source_size;
	fp = fopen("kernel1.cl", "r");
	if (!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		getchar();
		exit(1);
	}
	source_str = (char *)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	double total_time = 0;

	for (int testNum = 0; testNum < 10; testNum++)
	{
		cl_platform_id platform_id = NULL;
		cl_device_id device_id = NULL;

		cl_int ret = clGetPlatformIDs(1, &platform_id, NULL);
		ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
							 NULL);

		cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL,
											 &ret);

		clock_t t;

		clock_t end;

		cl_command_queue command_queue = clCreateCommandQueue(context,
															  device_id,
															  NULL, &ret);

		cl_mem doc_mem_obj = clCreateBuffer(context,
											CL_MEM_READ_ONLY, textLength * sizeof(char), NULL, &ret);
		cl_mem k_mem_obj = clCreateBuffer(context,
										  CL_MEM_READ_ONLY, patternLength * sizeof(char), NULL, &ret);
		cl_mem gs_mem_obj = clCreateBuffer(context,
										   CL_MEM_READ_ONLY, patternLength * sizeof(int), NULL, &ret);
		cl_mem bs_mem_obj = clCreateBuffer(context,
										   CL_MEM_READ_ONLY, 128 * sizeof(int), NULL, &ret);

		cl_mem se_mem_obj = clCreateBuffer(context,
										   CL_MEM_READ_ONLY, (numberOfProcesses * 2) * sizeof(int), NULL, &ret);

		cl_mem ans_mem_obj = clCreateBuffer(context,
											CL_MEM_READ_WRITE, (numberOfProcesses) * sizeof(int), NULL, &ret);

		ret = clEnqueueWriteBuffer(command_queue, doc_mem_obj, CL_TRUE, 0, textLength * sizeof(char), A, 0, NULL, NULL);

		int sth = str2.length();
		ret = clEnqueueWriteBuffer(command_queue, k_mem_obj, CL_TRUE, 0, sth * sizeof(char), str2.c_str(), 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue, gs_mem_obj, CL_TRUE, 0, patternLength * sizeof(int), goodSymTab, 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue, bs_mem_obj, CL_TRUE, 0, 128 * sizeof(int), badSymTab, 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue, se_mem_obj, CL_TRUE, 0, (numberOfProcesses * 2) * sizeof(int), start_endi, 0, NULL, NULL);

		cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

		ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
		clock_t begin = clock();
		cl_kernel kernel = clCreateKernel(program, "search", &ret);

		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&doc_mem_obj);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&k_mem_obj);
		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&se_mem_obj);
		ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&ans_mem_obj);
		ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&gs_mem_obj);
		ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&bs_mem_obj);
		ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&sth);

		size_t global_item_size = numberOfProcesses;
		size_t local_item_size = 1;

		cl_event event;
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1,
									 NULL, &global_item_size, &local_item_size, 0, NULL, &event);

		int ans[10];
		ret = clEnqueueReadBuffer(command_queue, ans_mem_obj, CL_TRUE, 0, numberOfProcesses * sizeof(int), ans, 0, NULL, NULL);
		end = clock();
		double time_spent = 0.0;
		time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

		total_time += time_spent;

		for (int x = 0; x < numberOfProcesses; x++)
			printf("\nThe no. of occurrences by process %d is %d", x, ans[x]);

		std::cout << "Time Spent:" << time_spent;

		clFlush(command_queue);

		ret = clReleaseKernel(kernel);
		ret = clReleaseProgram(program);

		ret = clReleaseMemObject(doc_mem_obj);
		ret = clReleaseMemObject(se_mem_obj);
		ret = clReleaseMemObject(k_mem_obj);
		ret = clReleaseMemObject(ans_mem_obj);
		ret = clReleaseMemObject(gs_mem_obj);
		ret = clReleaseMemObject(bs_mem_obj);
		ret = clReleaseCommandQueue(command_queue);
		ret = clReleaseContext(context);
	}
	double averageParallelTime = total_time / 10;
	std::cout << "Average time  = " << averageParallelTime << "\n";
}
