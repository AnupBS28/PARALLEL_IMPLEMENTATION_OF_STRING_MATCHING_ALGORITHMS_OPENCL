#define MAX_LOCAL_SIZE 256

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable






struct suffix
{
	int index; 
	int rank[2]; 
};

int cmp_constant_global(const __global struct suffix* a,const __global struct suffix* b)
{
	return (a->rank[0] == b->rank[0]) ? (a->rank[1] <= b->rank[1] ? 1 : 0) :(a->rank[0] < b->rank[0] ? 1 : 0);
}
int cmp_local(__local struct suffix *a, __local struct suffix *b)
{
	return (a->rank[0] == b->rank[0]) ? (a->rank[1] <= b->rank[1] ? 1 : 0) :(a->rank[0] < b->rank[0] ? 1 : 0);
}

int cmp(struct suffix *a, struct suffix *b)
{
	return (a->rank[0] == b->rank[0]) ? (a->rank[1] <= b->rank[1] ? 1 : 0) :(a->rank[0] < b->rank[0] ? 1 : 0);
}

inline void swap(struct suffix *a, struct suffix *b) {
	struct suffix tmp;
	tmp = *b;
	*b = *a;
	*a = tmp;
}

// dir == 1 means ascending
inline void sort(struct suffix *a, struct suffix *b, char dir) {
	if (cmp(a , b) == dir) swap(a, b);
}

inline void swapLocal(__local struct suffix *a, __local struct suffix *b) {
	struct suffix tmp;
	tmp = *b;
	*b = *a;
	*a = tmp;
}

// dir == 1 means ascending
inline void sortLocal(__local struct suffix *a, __local struct suffix *b, char dir) {
	if (cmp_local(a , b) == dir) swapLocal(a, b);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// basic kernel for mergesort start
__kernel void Sort_MergesortStart(const __global struct suffix* inArray, __global struct suffix* outArray)
{
	__local struct suffix local_buffer[2][MAX_LOCAL_SIZE * 2];
	const uint lid = get_local_id(0);
	const uint index = get_group_id(0) * (MAX_LOCAL_SIZE * 2) + lid;
	char pong = 0;
	char ping = 1;

	// load into local buffer
	local_buffer[0][lid] = inArray[index];
	local_buffer[0][lid + MAX_LOCAL_SIZE] = inArray[index + MAX_LOCAL_SIZE];

	// merge sort
	for (unsigned int stride = 2; stride <= MAX_LOCAL_SIZE * 2; stride <<= 1) {
		ping = pong;
		pong = 1 - ping;
		uint leftBoundary = lid * stride;
		uint rightBoundary = leftBoundary + stride;

		uint middle = leftBoundary + (stride >> 1);
		uint left = leftBoundary, right = middle;
		barrier(CLK_LOCAL_MEM_FENCE);
		if (rightBoundary > MAX_LOCAL_SIZE * 2) continue;
#pragma unroll
		for (uint i = 0; i < stride; i++) {
			struct suffix leftVal = local_buffer[ping][left];
			struct suffix rightVal = local_buffer[ping][right];
			bool selectLeft = left < middle && (right >= rightBoundary || cmp(&leftVal , &rightVal));

			local_buffer[pong][leftBoundary + i] = (selectLeft) ? leftVal : rightVal;

			left += selectLeft;
			right += 1 - selectLeft;
		}
	}

	//write back
	barrier(CLK_LOCAL_MEM_FENCE);
	outArray[index] = local_buffer[pong][lid];
	outArray[index + MAX_LOCAL_SIZE] = local_buffer[pong][lid + MAX_LOCAL_SIZE];
}

// For smaller strides so we can use local_buffer without getting into memory problems
__kernel void Sort_MergesortGlobalSmall(const __global struct suffix* inArray, __global struct suffix* outArray, const uint stride, const uint size)
{
	__local struct suffix local_buffer[MAX_LOCAL_SIZE * 2];

	// within one stride merge the different parts
	const uint baseIndex = get_global_id(0) * stride;
	const uint baseLocalIndex = get_local_id(0) * 2;

	uint middle = baseIndex + (stride >> 1);
	uint left = baseIndex;
	uint right = middle;
	bool selectLeft = false;

	if ((baseIndex + stride) > size) return;

	local_buffer[baseLocalIndex + 1] = inArray[left];

#pragma unroll
	for (uint i = baseIndex; i < (baseIndex + stride); i++) {
		// check which value should be written out
		local_buffer[baseLocalIndex + (int)selectLeft] = (selectLeft) ? inArray[left] : inArray[right];
		selectLeft = left < middle && (right == (baseIndex + stride) || cmp_local(&local_buffer[baseLocalIndex + 1] , &local_buffer[baseLocalIndex]));

		// write out
		outArray[i] = (selectLeft) ? local_buffer[baseLocalIndex + 1] : local_buffer[baseLocalIndex]; //PROBLEMATIC PART! WE RUN OUT OF MEMORY

		//increase counter accordingly
		left += selectLeft;
		right += 1 - selectLeft;
	}
}

__kernel void Sort_MergesortGlobalBig(const __global struct suffix* inArray, __global struct suffix* outArray, const uint stride, const uint size)
{
	//Problems: Breaks at large arrays. this version was stripped down (so little less performance but supports little bigger arrays)

	// within one stride merge the different parts
	const uint baseIndex = get_global_id(0) * stride;
	const char dir = 1;

	uint middle = baseIndex + (stride >> 1);
	uint left = baseIndex;
	uint right = middle;
	bool selectLeft;

	if ((baseIndex + stride) > size) return;

#pragma unroll
	for (uint i = baseIndex; i < (baseIndex + stride); i++) {
		// check which value should be written out
		selectLeft = (left < middle && (right == (baseIndex + stride) || cmp_constant_global(&inArray[left] , &inArray[right]))) == dir;

		// write out
		outArray[i] = (selectLeft) ? inArray[left] : inArray[right];

		//increase counter accordingly
		left += selectLeft;
		right += 1 - selectLeft;
	}
}


__kernel void init (__global struct suffix *suffixes,__global char *txt,__global int *n)
{
  int id=get_global_id(0);
	if(id<n[0]){
      suffixes[id].index = id;
		suffixes[id].rank[0] = txt[id] - 'a';
		suffixes[id].rank[1] = ((id + 1) < n[0]) ? (txt[id + 1] - 'a') : -1;
 }
}

__kernel void rank_to_suffix (__global struct suffix *suffixes,__global int *indices,const uint n,const uint k)
{			int i=get_global_id(0);
			if(i<n){
			int nextindex = suffixes[i].index + k;
			suffixes[i].rank[1] = (nextindex < n) ?suffixes[indices[nextindex]].rank[0] : -1;
	}
	//sort(suffixes, suffixes + n, cmp);
}


__kernel void later (__global struct suffix *suffixes,__global int *indices,__global int *n)
{			int i=get_global_id(0);
			if(i<n[0]){
	indices[i] = suffixes[i].index;
	}
}
