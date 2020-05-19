#include <stdio.h>
#include <CL/cl2.hpp>
#include<stdlib.h>
#include <cstdlib>
#pragma warning(disable: 4996)
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
	int index; // To store original index 
	int rank[2]; // To store ranks and next rank pair 
};
int cmp(struct suffix a, struct suffix b)
{
	return (a.rank[0] == b.rank[0]) ? (a.rank[1] < b.rank[1] ? 1 : 0) :
		(a.rank[0] < b.rank[0] ? 1 : 0);
}
size_t GetGlobalWorkSize(size_t DataElemCount, size_t LocalWorkSize)
{
	size_t r = DataElemCount % LocalWorkSize;
	if (r == 0)
		return DataElemCount;
	else
		return DataElemCount + LocalWorkSize - r;
}
void Sort_Mergesort(cl_context Context, cl_command_queue CommandQueue, size_t LocalWorkSize[3], struct suffix* m_hInput, cl_kernel m_MergesortStartKernel, cl_kernel	m_MergesortGlobalSmallKernel, cl_kernel	m_MergesortGlobalBigKernel, cl_mem	m_dPingArray, cl_mem m_dPongArray, size_t m_N_padded)
{
	//TODO fix memory problem when many elements. -> CL_OUT_OF_RESOURCES
	cl_int clError;
	size_t globalWorkSize[1];
	size_t localWorkSize[1];

	localWorkSize[0] = LocalWorkSize[0];
	globalWorkSize[0] = GetGlobalWorkSize(m_N_padded / 2, localWorkSize[0]);
	unsigned int locLimit = 1;
	clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N_padded * sizeof(struct suffix), m_hInput, 0, NULL, NULL);
	if (m_N_padded >= LocalWorkSize[0] * 2) {
		locLimit = 2 * LocalWorkSize[0];

		// start with a local variant first, ASSUMING we have more than localWorkSize[0] * 2 elements
		clError = clSetKernelArg(m_MergesortStartKernel, 0, sizeof(cl_mem), (void*)&m_dPingArray);
		clError |= clSetKernelArg(m_MergesortStartKernel, 1, sizeof(cl_mem), (void*)&m_dPongArray);
		//V_RETURN_CL(clError, "Failed to set kernel args: MergeSortStart");

		clError = clEnqueueNDRangeKernel(CommandQueue, m_MergesortStartKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		//V_RETURN_CL(clError, "Error executing MergeSortStart kernel!");
		//clEnqueueReadBuffer(CommandQueue, m_dPongArray, CL_TRUE, 0, m_N_padded * sizeof(struct suffix), m_hInput, 0, NULL, NULL);
		swap(m_dPingArray, m_dPongArray);
	}
	//clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N_padded * sizeof(struct suffix), m_hInput, 0, NULL, NULL);
	// proceed with the global variant
	unsigned int stride = 2 * locLimit;

	localWorkSize[0] = LocalWorkSize[0];
	globalWorkSize[0] = GetGlobalWorkSize(m_N_padded / 2, localWorkSize[0]);

	if (m_N_padded <= MERGESORT_SMALL_STRIDE) {
		// set not changing arguments
		clError = clSetKernelArg(m_MergesortGlobalSmallKernel, 3, sizeof(cl_uint), (void*)&m_N_padded);
		//V_RETURN_CL(clError, "Failed to set kernel args: MergeSortGlobal");

		for (; stride <= m_N_padded; stride <<= 1) {
			//calculate work sizes
			size_t neededWorkers = m_N_padded / stride;
			//clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N_padded * sizeof(struct suffix), m_hInput, 0, NULL, NULL);
			localWorkSize[0] = min(LocalWorkSize[0], neededWorkers);
			globalWorkSize[0] = GetGlobalWorkSize(neededWorkers, localWorkSize[0]);

			clError = clSetKernelArg(m_MergesortGlobalSmallKernel, 0, sizeof(cl_mem), (void*)&m_dPingArray);
			clError |= clSetKernelArg(m_MergesortGlobalSmallKernel, 1, sizeof(cl_mem), (void*)&m_dPongArray);
			clError |= clSetKernelArg(m_MergesortGlobalSmallKernel, 2, sizeof(cl_uint), (void*)&stride);
			//V_RETURN_CL(clError, "Failed to set kernel args: MergeSortGlobal");

			clError = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalSmallKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
			//V_RETURN_CL(clError, "Error executing kernel!");

			swap(m_dPingArray, m_dPongArray);
			//clEnqueueReadBuffer(CommandQueue, m_dPongArray, CL_TRUE, 0, m_N_padded * sizeof(struct suffix), m_hInput, 0, NULL, NULL);
			/*for (int yu = 0;yu < m_N_padded;yu++)
			{
				printf("%d %d %d m_hInput \n", m_hInput[yu].rank[0], m_hInput[yu].rank[1], m_hInput[yu].index);
			}*/
		}
	}
	else {
		// set not changing arguments
		clError = clSetKernelArg(m_MergesortGlobalBigKernel, 3, sizeof(cl_uint), (void*)&m_N_padded);
		//V_RETURN_CL(clError, "Failed to set kernel args: MergeSortGlobal");

		for (; stride <= m_N_padded; stride <<= 1) {
			//calculate work sizes
			size_t neededWorkers = m_N_padded / stride;
			//clEnqueueWriteBuffer(CommandQueue, m_dPingArray, CL_FALSE, 0, m_N_padded * sizeof(struct suffix), m_hInput, 0, NULL, NULL);
			localWorkSize[0] = min(LocalWorkSize[0], neededWorkers);
			globalWorkSize[0] = GetGlobalWorkSize(neededWorkers, localWorkSize[0]);

			clError = clSetKernelArg(m_MergesortGlobalBigKernel, 0, sizeof(cl_mem), (void*)&m_dPingArray);
			clError |= clSetKernelArg(m_MergesortGlobalBigKernel, 1, sizeof(cl_mem), (void*)&m_dPongArray);
			clError |= clSetKernelArg(m_MergesortGlobalBigKernel, 2, sizeof(cl_uint), (void*)&stride);
			//V_RETURN_CL(clError, "Failed to set kernel args: MergeSortGlobal");

			clError = clEnqueueNDRangeKernel(CommandQueue, m_MergesortGlobalBigKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
			//V_RETURN_CL(clError, "Error executing kernel!");

			//if (stride >= 1024 * 1024) V_RETURN_CL(clFinish(CommandQueue), "Failed finish CommandQueue at mergesort for bigger strides.");
			swap(m_dPingArray, m_dPongArray);
			//clEnqueueReadBuffer(CommandQueue, m_dPongArray, CL_TRUE, 0, m_N_padded * sizeof(struct suffix), m_hInput, 0, NULL, NULL);
		}
	}
	clEnqueueReadBuffer(CommandQueue, m_dPingArray, CL_TRUE, 0, m_N_padded * sizeof(struct suffix), m_hInput, 0, NULL, NULL);
	//return m_hInput;
}


int* buildSuffixArray(char* txt, int n)
{
	// A structure to store suffixes and their indexes 
	struct suffix* suffixes;
	suffixes = (struct suffix*)malloc(sizeof(struct suffix) * n);

	// Store suffixes and their indexes in an array of structures. 
	// The structure is needed to sort the suffixes alphabatically 
	// and maintain their old indexes while sorting 
	for (int i = 0; i < n; i++)
	{
		suffixes[i].index = i;
		suffixes[i].rank[0] = txt[i] - 'a';
		suffixes[i].rank[1] = ((i + 1) < n) ? (txt[i + 1] - 'a') : -1;
	}

	// Sort the suffixes using the comparison function 
	// defined above. 
	sort(suffixes, suffixes + n, cmp);

	// At this point, all suffixes are sorted according to first 
	// 2 characters.  Let us sort suffixes according to first 4 
	// characters, then first 8 and so on 
	int* ind;  // This array is needed to get the index in suffixes[] 
	ind = (int*)malloc(sizeof(int) * n);
	// from original index.  This mapping is needed to get 
	  // next suffix. 
	for (int k = 4; k < 2 * n; k = k * 2)
	{
		// Assigning rank and index values to first suffix 
		int rank = 0;
		int prev_rank = suffixes[0].rank[0];
		suffixes[0].rank[0] = rank;
		ind[suffixes[0].index] = 0;

		// Assigning rank to suffixes 
		for (int i = 1; i < n; i++)
		{
			// If first rank and next ranks are same as that of previous 
			// suffix in array, assign the same new rank to this suffix 
			if (suffixes[i].rank[0] == prev_rank &&
				suffixes[i].rank[1] == suffixes[i - 1].rank[1])
			{
				prev_rank = suffixes[i].rank[0];
				suffixes[i].rank[0] = rank;
			}
			else // Otherwise increment rank and assign 
			{
				prev_rank = suffixes[i].rank[0];
				suffixes[i].rank[0] = ++rank;
			}
			ind[suffixes[i].index] = i;
		}

		// Assign next rank to every suffix 
		for (int i = 0; i < n; i++)
		{
			int nextindex = suffixes[i].index + k / 2;
			suffixes[i].rank[1] = (nextindex < n) ?
				suffixes[ind[nextindex]].rank[0] : -1;
		}

		// Sort the suffixes according to first k characters 
		sort(suffixes, suffixes + n, cmp);
	}

	// Store indexes of all sorted suffixes in the suffix array 
	int* suffixArr = new int[n];
	for (int i = 0; i < n; i++)
		suffixArr[i] = suffixes[i].index;

	// Return the suffix array 
	return  suffixArr;
}




void printArr(int arr[], int n)
{
	for (int i = 0; i < n; i++)
		cout << arr[i] << " ";
	cout << endl;
}

int main(int argc, char** argv)
{
	//int np = 2;
	//int i = 0, j = 0, leng, nword = 0;

	/*std::ifstream MyReadFile("tests.txt");
	std::string line;

	getline(MyReadFile, line);
	MyReadFile.close();

	std::ofstream ofs("tests.txt", std::ofstream::trunc);
	int fileNumber = stoi(line);
	fileNumber++;
	ofs << fileNumber;
	ofs.close();*/
	/*
	std::ifstream ifs("inputEd.txt");
	std::string str1((std::istreambuf_iterator<char>(ifs)),
		(std::istreambuf_iterator<char>()));

	std::ifstream ifs2("in.txt");
	std::string str2((std::istreambuf_iterator<char>(ifs2)),
		(std::istreambuf_iterator<char>()));
	leng = str1.length();
	flen = str2.length();
	*/
	/*char str[100], B[10];
	printf("Enter the string: ");
	gets_s(str);
	leng = strlen(str);
	printf("Enter the word to find:");
	gets_s(B);
	flen = strlen(B);*/
	//char* A = (char*)malloc(sizeof(char) * (leng + 1));
	//strcpy(A, str1.c_str());

	cl_kernel			m_MergesortStartKernel;
	cl_kernel			m_MergesortGlobalSmallKernel;
	cl_kernel			m_MergesortGlobalBigKernel;

	cl_mem				m_dPingArray;
	cl_mem				m_dPongArray;
	size_t				m_N;
	size_t				m_N_padded;
	//size_t				LocalWorkSize[3];
	size_t LocalWorkSize[3] = { 256, 1 , 1 };
	//std::cout << A;
	FILE* infile1, * infile2;
	char* a, * b;
	char* program_log;
	//long sizea, sizeb;

	infile1 = fopen("input.txt", "r");
	//infile2 = fopen("str2.txt", "r");

	if (infile1 == NULL) {
		printf("File Not Found!\n");
		return -1;
	}

	fseek(infile1, 0L, SEEK_END);
	long int n = ftell(infile1);



	fseek(infile1, 0L, SEEK_SET);




	//long int str1len = n;
	//long int str2len = sizeb;


	/*std::ifstream ifs("input.txt");
	std::string str1((std::istreambuf_iterator<char>(ifs)),
		(std::istreambuf_iterator<char>()));
		*/
	clock_t t;
	clock_t begin;
	clock_t end;

	//char txt[] = "bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkjhehfdshsjkhfkjshsjdhfuehfsdfsdkjfheuedjfkjashfj bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf bananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjfbananagsjkj sdfsjf hfjsahf sd fshl flsfhs flskfdh shfj sdfhsl fsdfhs fsfsfljsf dsf fsjf sfshfeus dffjsdfh sflsdf slfs fslfjsd flsfjsh fls fsfhfhs flsf fsh lfsd flsdf sf sdl flsfs dfsdlf sdlfsf sdfd flfs f sfjsflsjf kjsdfhjshfjshfsjh jsdfhjhsd sfjdfl sfhsdjfh dsjfh sfjhsdl fsdfhsjf sdkfhsdfh sfsf sdfl sdlfjsdf dlfj sdflsdfh dfdf dsfld fsdf slfsfjf sdfldsfhsd fdsfjd fjdsf slfsdflfh sffldsf sfsfh dfsjf sf sdf sdlf shdfjh fjdfh sdjhfsld fhdsf sflds fdhsd fhdslf sfsh fsf fd slfsf sdfhsdf sdfhsf sjsdfls sdjfs jdsfsdfh fjsfsdhjfh sfl dsf lsfdjh djs jfds dsfhjh sjl fdfhsd dfjsl lsf sf slfsh fsj fhs flsdfsld fsf sl dsl fsd fsld slf sl lasf lla lf as ls fs fsal sla fdfsf fhfjs eyrwe lhfflas fs ncbxnznbvkjeiwoie fh wejfhdsjfh jhjdhf sdfhjsd hldjhf fjdhf sdfhjd fhsj j hfhsjfhs jfsljd s hkjhf lsjf jfdsjdfkjsdkfjskjfsfjskfjskfjsjdfsfh fjsdhf jhf sdhfs ahfh kfs  sl sdhf dsh fdfjflasj djfhsf jsdhf djfsfh sdjhfjsdl jhfjshf jhfhsla kjjhjksa lshfs jkhf jhfjsh fudruej dfhfhudfj lsahfjhd i jsdfl dfhjhds kjhjf kjfj jkhfjhls fkjsjfheuuehdshfjfdsjf kfsdjfsdl fsjfksj flsdkfj slkdfjlskdjf lsjdfsdlkj flsdkf jsl fjsflsk fslkfj sdf dhfuierfhjsdjfhsjfh sjfhief sjdsjdf klsjfslfjflsdjf ei jdsfkj slkfjskj jsdjhfjdslhfjsfsdhkj";
	//int n = strlen(txt);
	//int n=str1.length();
	cout << n << endl;
	char* txt;
	txt = (char*)calloc(n, sizeof(char));


	if (txt == NULL)
		return 1;


	fread(txt, sizeof(char), n, infile1);
	fclose(infile1);
	//char* txt = (char*)malloc(sizeof(char) * (n+ 1));
	//cout << txt << endl;
	//strcpy(txt, str1.c_str());

	struct suffix* suffixes;
	suffixes = (struct suffix*)malloc(sizeof(struct suffix) * n);
	begin = clock();
	int* sequential_array = buildSuffixArray(txt, n);
	end = clock();
	cout << "sequential  array output :" << n << endl;
	//printArr(sequential_array, n);

	double  ti = 0.0;
	ti = (double)(end - begin) / CLOCKS_PER_SEC;

	std::cout << "sequential time spent:" << ti << endl;

	/*	int* suffixArr = buildSuffixArray(txt, n);
		cout << "Following is suffix array for " << txt << endl;
		printArr(suffixArr, n);
		*/



		// Load the kernel source code into the array source_str
	FILE* fp;
	char* source_str;
	size_t source_size;
	fp = fopen("kernel.cl", "r");
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
	if (ret < 0) {
		printf("pm po %d", ret);
		exit(1);
	}
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
		NULL);
	// Create an OpenCL context
	if (ret < 0) {
		printf("pm rt %d", ret);
		exit(1);
	}
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL,
		&ret);
	if (ret < 0) {
		printf("pm bner %d", ret);
		exit(1);
	}
	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context,
		device_id,
		NULL, &ret);
	// Create buffers
	cl_mem m_txt = clCreateBuffer(context,
		CL_MEM_READ_ONLY, n * sizeof(char), NULL, &ret);
	if (ret < 0) {
		printf("pm op %d", ret);
		exit(1);
	}
	cl_mem m_suffixes = clCreateBuffer(context,
		CL_MEM_READ_WRITE, n * sizeof(struct suffix), NULL, &ret);
	if (ret < 0) {
		printf("pm ml %d", ret);
		exit(1);
	}
	cl_mem m_n = clCreateBuffer(context,
		CL_MEM_READ_ONLY, 1 * sizeof(int), NULL, &ret);
	if (ret < 0) {
		printf("pm gh %d", ret);
		exit(1);
	}
	cl_mem m_k = clCreateBuffer(context,
		CL_MEM_READ_ONLY, 1 * sizeof(int), NULL, &ret);
	if (ret < 0) {
		printf("pm gh %d", ret);
		exit(1);
	}
	cl_mem m_indices = clCreateBuffer(context,
		CL_MEM_READ_ONLY, n * sizeof(int), NULL, &ret);
	if (ret < 0) {
		printf("pm gh %d", ret);
		exit(1);
	}
	cl_int clError, clError2;
	m_dPingArray = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(struct suffix) * n, NULL, &clError2);
	ret = clError2;
	m_dPongArray = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(struct suffix) * n, NULL, &clError2);
	ret |= clError2;
	if (ret < 0) {
		printf("pm ghry %d", ret);
		exit(1);
	}

	ret = clEnqueueWriteBuffer(command_queue, m_txt, CL_TRUE, 0, n * sizeof(char), txt, 0, NULL, NULL);


	//ret = clEnqueueWriteBuffer(command_queue, m_suffixes, CL_TRUE, 0, n * sizeof(struct suffix), suffixes, 0, NULL, NULL);
	ret |= clEnqueueWriteBuffer(command_queue, m_n, CL_TRUE, 0, 1 * sizeof(int), &n, 0, NULL, NULL);
	if (ret < 0) {
		printf("pm qe %d", ret);
		exit(1);
	}
	//step 7
	cl_program program = clCreateProgramWithSource(context, 1, (const
		char**)&source_str, (const size_t*)&source_size, &ret);
	if (ret < 0) {
		printf("pm n %d", ret);
		exit(1);
	}
	size_t  log_size;
	ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (ret < 0) {
		//printf("pm jk %d", ret);
		//exit(1);
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
			0, NULL, &log_size);
		program_log = (char*)malloc(log_size + 1);
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
			log_size + 1, program_log, NULL);
		printf("%s\n", program_log);
		free(program_log);
		exit(1);
	}

	cl_kernel kernel = clCreateKernel(program, "init", &ret);
	if (ret < 0) {
		printf("pm bn %d", ret);
		exit(1);
	}
	cl_kernel rank_to_suffix = clCreateKernel(program, "rank_to_suffix", &ret);
	if (ret < 0) {
		printf("pm bn2 %d", ret);
		exit(1);
	}
	cl_kernel later = clCreateKernel(program, "later", &ret);
	if (ret < 0) {
		printf("pm bn2 %d", ret);
		exit(1);
	}

	m_MergesortGlobalSmallKernel = clCreateKernel(program, "Sort_MergesortGlobalSmall", &clError);
	//V_RETURN_FALSE_CL(clError, "Failed to create kernel: Sort_MergesortGlobalSmall.");
	ret = clError;
	m_MergesortGlobalBigKernel = clCreateKernel(program, "Sort_MergesortGlobalBig", &clError);
	ret |= clError;
	//V_RETURN_FALSE_CL(clError, "Failed to create kernel: Sort_MergesortGlobalBig.");
	m_MergesortStartKernel = clCreateKernel(program, "Sort_MergesortStart", &clError);
	ret |= clError;
	if (ret < 0) {
		printf("pm bn2 %d", ret);
		exit(1);
	}
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&m_suffixes);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&m_txt);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&m_n);
	if (ret < 0) {
		printf("pm qw %d", ret);
		exit(1);
	}


	if (ret < 0) {
		printf("pm qw %d", ret);
		exit(1);
	}

	ret = clSetKernelArg(later, 0, sizeof(cl_mem), (void*)&m_suffixes);
	ret |= clSetKernelArg(later, 1, sizeof(cl_mem), (void*)&m_indices);
	ret |= clSetKernelArg(later, 2, sizeof(cl_mem), (void*)&m_n);

	if (ret < 0) {
		printf("pm qw %d", ret);
		exit(1);
	}

	size_t local_item_size = 256;
	size_t global_item_size = GetGlobalWorkSize(n, local_item_size);

	//int n = strlen(txt);
	begin = clock();
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

	ret = clEnqueueReadBuffer(command_queue, m_suffixes, CL_TRUE, 0, n * sizeof(struct suffix), suffixes, 0, NULL, NULL);

	//struct suffix* test_sort;
	//test_sort = (struct suffix*)malloc(sizeof(struct suffix) * n);
	//memcpy(test_sort,&suffixes[0], sizeof(struct suffix)* n);
	// Store suffixes and their indexes in an array of structures. 
	// The structure is needed to sort the suffixes alphabatically 
	// and maintain their old indexes while sorting 
	/*for (int i = 0; i < n; i++)
	{
		//printf("%d %d %d \n", suffixes[i].index, suffixes[i].rank[0], suffixes[i].rank[1]);
		suffixes[i].index = i;
		suffixes[i].rank[0] = txt[i] - 'a';
		suffixes[i].rank[1] = ((i + 1) < n) ? (txt[i + 1] - 'a') : -1;
	}*/

	// Sort the suffixes using the comparison function 
	// defined above. 
	//sort(suffixes, suffixes + n, cmp);
	Sort_Mergesort(context, command_queue, LocalWorkSize, suffixes, m_MergesortStartKernel, m_MergesortGlobalSmallKernel, m_MergesortGlobalBigKernel, m_dPingArray, m_dPongArray, n);
	// At this point, all suffixes are sorted according to first 
	// 2 characters.  Let us sort suffixes according to first 4 
	// characters, then first 8 and so on 

	//sort(test_sort, test_sort + n, cmp);
	//int ryu = 0;
	//for (int yu = 0;yu < n;yu++)
	//{
	//	//printf("%d %d %d\n", suffixes[yu].rank[0], suffixes[yu].rank[1], suffixes[yu].index);
	//	if (test_sort[yu].rank[0] != suffixes[yu].rank[0] || test_sort[yu].rank[1] != suffixes[yu].rank[1] )
	//	{
	//		ryu ++;

	//		printf("%d %d %d %d %d %d\n", test_sort[yu].rank[0], test_sort[yu].rank[1], test_sort[yu].index, suffixes[yu].rank[0], suffixes[yu].rank[1], suffixes[yu].index);
	//		//printf("%d io\n", yu);
	//		if(ryu==1000)
	//			break;
	//	}
	//}
	//if (ryu==0)
	//{
	//	printf("same\n");
	//}
	//else
	//	printf("not same\n");
	int* ind;  // This array is needed to get the index in suffixes[] 
	ind = (int*)malloc(sizeof(int) * n);
	// from original index.  This mapping is needed to get 
	  // next suffix. 
//ret = clSetKernelArg(rank_to_suffix, 0, sizeof(cl_mem), (void*)&m_suffixes);
//ret |= clSetKernelArg(rank_to_suffix, 1, sizeof(cl_mem), (void*)&m_indices);
//ret |= clSetKernelArg(rank_to_suffix, 3, sizeof(cl_mem), (void*)&m_k);
//ret |= clEnqueueWriteBuffer(command_queue, m_n, CL_TRUE, 0, 1 * sizeof(int), &n, 0, NULL, NULL);
//ret |= clEnqueueWriteBuffer(command_queue, m_k, CL_TRUE, 0, 1 * sizeof(int), &y, 0, NULL, NULL);
//ret |= clEnqueueWriteBuffer(command_queue, m_indices, CL_TRUE, 0, n * sizeof(int), ind, 0, NULL, NULL);
//ret |= clSetKernelArg(rank_to_suffix, 2, sizeof(cl_mem), (void*)&m_n);
	unsigned int hj = n;
	ret = clSetKernelArg(rank_to_suffix, 2, sizeof(cl_uint), (void*)&hj);
	/*if (ret < 0) {
		printf("pm qetw %d", ret);
		exit(1);
	}*/
	for (int k = 4; k < 2 * n; k = k * 2)
	{
		// Assigning rank and index values to first suffix 
		int rank = 0;
		int prev_rank = suffixes[0].rank[0];
		suffixes[0].rank[0] = rank;
		ind[suffixes[0].index] = 0;
		unsigned int y = k / 2;
		ret |= clSetKernelArg(rank_to_suffix, 3, sizeof(cl_uint), (void*)&y);
		/*if (ret < 0) {
			printf("pm qqrtw %d", ret);
			exit(1);
		}*/
		// Assigning rank to suffixes 
		for (int i = 1; i < n; i++)
		{
			// If first rank and next ranks are same as that of previous 
			// suffix in array, assign the same new rank to this suffix 
			if (suffixes[i].rank[0] == prev_rank &&
				suffixes[i].rank[1] == suffixes[i - 1].rank[1])
			{
				prev_rank = suffixes[i].rank[0];
				suffixes[i].rank[0] = rank;
			}
			else // Otherwise increment rank and assign 
			{
				prev_rank = suffixes[i].rank[0];
				suffixes[i].rank[0] = ++rank;
			}
			ind[suffixes[i].index] = i;
		}

		// Assign next rank to every suffix 
		//for (int i = 0; i < n; i++)
		//{
			////printf("%d %d %d before\n", suffixes[i].rank[0], suffixes[i].rank[1], suffixes[i].index);
			//int nextindex = suffixes[i].index + k / 2;
			//suffixes[i].rank[1] = (nextindex < n) ?suffixes[ind[nextindex]].rank[0] : -1;

		//}
		//ret |= clEnqueueWriteBuffer(command_queue, m_n, CL_TRUE, 0, 1 * sizeof(int), &n, 0, NULL, NULL);
		//ret |= clEnqueueWriteBuffer(command_queue, m_k, CL_TRUE, 0, 1 * sizeof(int), &y, 0, NULL, NULL);
		ret |= clEnqueueWriteBuffer(command_queue, m_indices, CL_TRUE, 0, n * sizeof(int), ind, 0, NULL, NULL);

		ret |= clEnqueueWriteBuffer(command_queue, m_suffixes, CL_TRUE, 0, n * sizeof(struct suffix), suffixes, 0, NULL, NULL);
		//ret |= clEnqueueWriteBuffer(command_queue, m_indices, CL_TRUE, 0, n * sizeof(int), ind, 0, NULL, NULL);

		ret |= clSetKernelArg(rank_to_suffix, 0, sizeof(cl_mem), (void*)&m_suffixes);
		ret |= clSetKernelArg(rank_to_suffix, 1, sizeof(cl_mem), (void*)&m_indices);

		//ret |= clSetKernelArg(rank_to_suffix, 3, sizeof(cl_mem), (void*)&m_k);
		// Sort the suffixes according to first k characters
		/*if (ret < 0) {
			printf("pm qw %d", ret);
			exit(1);
		}*/

		ret = clEnqueueNDRangeKernel(command_queue, rank_to_suffix, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

		ret = clEnqueueReadBuffer(command_queue, m_suffixes, CL_TRUE, 0, n * sizeof(struct suffix), suffixes, 0, NULL, NULL);

		//for (int i = 0; i < n; i++)
		//{
			////int nextindex = suffixes[i].index + k / 2;
			////suffixes[i].rank[1] = (nextindex < n) ?suffixes[ind[nextindex]].rank[0] : -1;
			//printf("%d %d %d after\n", suffixes[i].rank[0], suffixes[i].rank[1], suffixes[i].index);
		//}x
		Sort_Mergesort(context, command_queue, LocalWorkSize, suffixes, m_MergesortStartKernel, m_MergesortGlobalSmallKernel, m_MergesortGlobalBigKernel, m_dPingArray, m_dPongArray, n);
		//sort(suffixes, suffixes + n, cmp);
	}

	//// Store indexes of all sorted suffixes in the suffix array 
	//ret = clEnqueueWriteBuffer(command_queue, m_suffixes, CL_TRUE, 0, n * sizeof(struct suffix), suffixes, 0, NULL, NULL);
	////ret |= clEnqueueWriteBuffer(command_queue, m_indices, CL_TRUE, 0, n * sizeof(int), ind, 0, NULL, NULL);
	int* suffixArr = new int[n];
	for (int i = 0; i < n; i++)
		suffixArr[i] = suffixes[i].index;
	//ret = clEnqueueNDRangeKernel(command_queue, later, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
	//ret = clEnqueueReadBuffer(command_queue, m_indices, CL_TRUE, 0, n * sizeof(int), suffixArr, 0, NULL, NULL);
	end = clock();
	//cout << "Following is suffix array for " << txt << endl;
	//printArr(suffixArr, n);
	int tik;
	tik = 0;
	for (int i = 0; i < n; i++) {
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
	double  time_spent = 0.0;
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