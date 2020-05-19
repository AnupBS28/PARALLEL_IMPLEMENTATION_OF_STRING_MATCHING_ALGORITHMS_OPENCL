#include <stdio.h>
#include <CL/cl2.hpp>
#include<stdlib.h>
#pragma warning(disable: 4996)
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>
#define MAX_SOURCE_SIZE 0x1000000

int flen;

int max(int a, int b)
{
	if (a > b) return a;
	else return b;
}

int matchfirst(char word[100], int sub)
{
	int i;
	int lensub = flen - sub;
	for (i = 0; i < lensub; i++)
	{
		if (word[i] != word[sub + i])
		{
			return -1;
		}

	}
	return 0;
}


int match(char word[100], int sub)
{
	int prev = word[sub - 1];
	int lensub = flen - sub;
	int j; int h;

	for (j = sub - 1; j >= 0; j--)
	{
		if (j - 1 >= 0)
		{
			if (prev == word[j - 1])continue;
		}
		int flag = 0; int count = 0;
		for (h = j; h <= j + lensub - 1; h++)
		{
			if (word[h] != word[sub + count])
			{
				flag = 1;
			}
			count++;
		}

		if (flag == 1)
			continue;
		else
			return j;

	}
	return -1;

}

int main(int argc, char** argv)
{
	int np = 2;
	int i = 0, j = 0, leng, nword = 0;

	/*std::ifstream MyReadFile("tests.txt");
	std::string line;

	getline(MyReadFile, line);
	MyReadFile.close();

	std::ofstream ofs("tests.txt", std::ofstream::trunc);
	int fileNumber = stoi(line);
	fileNumber++;
	ofs << fileNumber;
	ofs.close();*/

	std::ifstream ifs("inputEd.txt");
	std::string str1((std::istreambuf_iterator<char>(ifs)),
		(std::istreambuf_iterator<char>()));

	std::ifstream ifs2("input1Search.txt");
	std::string str2((std::istreambuf_iterator<char>(ifs2)),
		(std::istreambuf_iterator<char>()));
	leng = str1.length();
	flen = str2.length();

	/*char str[100], B[10];
	printf("Enter the string: ");
	gets_s(str);
	leng = strlen(str);
	printf("Enter the word to find:");
	gets_s(B);
	flen = strlen(B);*/
	char* A = (char*)malloc(sizeof(char) * (leng + 1));
	strcpy(A, str1.c_str());

	std::cout << A;


	while (A[i++] != '\0')
	{
		if (A[i] == ' ')
			nword++;

	}

	int* len = (int*)malloc(sizeof(int) * (nword + 1));
	for (i = 0; i < (nword + 1); i++)
		len[i] = 0;
	i = 0;
	while (j < nword + 1)
	{

		while (A[i] != ' ' && A[i] != '\0')
		{
			len[j]++;
			i++;
		}
		j++;
		i++;
	}

	int nwpp = (nword + 1) / np;
	//int [np * 2];
	int* start_endi = (int*)malloc(sizeof(int) * (np * 2));
	int count = 0; int si, ei; int k = 0; int lengthindex = 0;
	for (i = 0; i < np; i++)
	{
		si = count; ei = si;
		for (j = lengthindex; j < lengthindex + nwpp; j++)
		{

			ei += len[j];
		}
		lengthindex += nwpp;
		ei = ei + nwpp - 1;
		start_endi[k++] = si;
		start_endi[k++] = ei - 1;
		count = ei + 1;
	}

	/*for (i = 0; i<np * 2; i++)
	{
		printf("*******************%d  ", start_endi[i]);
	}*/
	int fl = 0;
	char word[100]; int f, sub, result;
	strcpy(word, str2.c_str());
	
	
	
	int dlen = str1.length();
	//int gstable[flen];
	int* gstable = (int*)malloc(sizeof(int) * (flen));
	int bstable[128];
	for (i = 0; i <= 127; i++)
	{
		bstable[i] = flen;
	}
	for (i = 0; i <= flen - 2; i++)
	{
		bstable[(int)word[i]] = flen - 1 - i;
	}

	for (k = 1; k <= flen - 1; k++)
	{
		sub = flen - k; f = 0;
		result = match(word, sub);
		if (result >= 0)
		{
			gstable[k] = sub - result;
			continue;
		}
		for (sub = flen - k + 1; sub <= flen - 1; sub++)
		{
			result = matchfirst(word, sub);
			if (result == 0)
			{
				gstable[k] = sub - result;
				f = 1;
				break;
			}
		}
		if (f == 0)
		{

			gstable[k] = flen;
		}
	}

	/*printf("\n\n*********BAD SYMBOL TABLE*********\n");
	for (i = 0; i <= 127; i++)
	{
		printf("\n%d ", bstable[i]);
	}
	printf("\n\n******GOOD SUFFIX TABLE IS **********\n");
	for (i = 1; i <= flen - 1; i++)
	{
		printf("\n%d\n", gstable[i]);
	}*/

	//////daa code ends

	// Load the kernel source code into the array source_str
	FILE* fp;
	char* source_str;
	size_t source_size;
	fp = fopen("kernel1.cl", "r");
	if (!fp)
	{
		fprintf(stderr, "Failed to load kernel.\n");
		getchar();
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
	// Get platform and device information
	cl_platform_id platform_id = NULL;
	cl_device_id device_id = NULL;
	//cl_uint ret_num_devices;
	//cl_uint ret_num_platforms;

	cl_int ret = clGetPlatformIDs(1, &platform_id, NULL);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
		NULL);
	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL,
		&ret);

	clock_t t;
	
	clock_t end;
	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context,
		device_id,
		NULL, &ret);
	// Create buffers
	cl_mem doc_mem_obj = clCreateBuffer(context,
		CL_MEM_READ_ONLY, leng * sizeof(char), NULL, &ret);
	cl_mem k_mem_obj = clCreateBuffer(context,
		CL_MEM_READ_ONLY, flen * sizeof(char), NULL, &ret);
	cl_mem gs_mem_obj = clCreateBuffer(context,
		CL_MEM_READ_ONLY, flen * sizeof(int), NULL, &ret);
	cl_mem bs_mem_obj = clCreateBuffer(context,
		CL_MEM_READ_ONLY, 128 * sizeof(int), NULL, &ret);

	cl_mem se_mem_obj = clCreateBuffer(context,
		CL_MEM_READ_ONLY, (np * 2) * sizeof(int), NULL, &ret);

	cl_mem ans_mem_obj = clCreateBuffer(context,
		CL_MEM_READ_WRITE, (np) * sizeof(int), NULL, &ret);


	ret = clEnqueueWriteBuffer(command_queue, doc_mem_obj, CL_TRUE, 0, leng * sizeof(char), A, 0, NULL, NULL);

	int sth = str2.length();
	ret = clEnqueueWriteBuffer(command_queue, k_mem_obj, CL_TRUE, 0, sth * sizeof(char), str2.c_str(), 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, gs_mem_obj, CL_TRUE, 0, flen * sizeof(int), gstable, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, bs_mem_obj, CL_TRUE, 0, 128 * sizeof(int), bstable, 0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, se_mem_obj, CL_TRUE, 0, (np * 2) * sizeof(int), start_endi, 0, NULL, NULL);
	//step 7
	cl_program program = clCreateProgramWithSource(context, 1, (const
		char**)&source_str, (const size_t*)&source_size, &ret);


	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	clock_t begin = clock();
	cl_kernel kernel = clCreateKernel(program, "search", &ret);


	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&doc_mem_obj);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&k_mem_obj);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&se_mem_obj);
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&ans_mem_obj);
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&gs_mem_obj);
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bs_mem_obj);
	ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&sth);


	size_t global_item_size = np;
	size_t local_item_size = 1;

	cl_event event;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1,
		NULL, &global_item_size, &local_item_size, 0, NULL, &event);



	int ans[10];
	ret = clEnqueueReadBuffer(command_queue, ans_mem_obj, CL_TRUE, 0, np * sizeof(int), ans, 0, NULL, NULL);
	end = clock();
	double  time_spent = 0.0;
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	for (int x = 0; x < np; x++)
		printf("\nThe no of occurrences by process %d is %d", x, ans[x]);

	

	std::ofstream output("output.txt", std::ios::app);
	output << time_spent<<"\n";

	std::cout << "time spent:" << time_spent;

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


