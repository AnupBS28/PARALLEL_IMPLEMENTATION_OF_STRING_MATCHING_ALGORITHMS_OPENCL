#define PROGRAM_FILE "kernal.cl"
#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

size_t GetGlobalWorkSize(size_t DataElemCount, size_t LocalWorkSize)
{
    size_t r = DataElemCount % LocalWorkSize;
    if (r == 0)
        return DataElemCount;
    else
        return DataElemCount + LocalWorkSize - r;
}
int main()
{

    /* Host/device data structures */
    int iterations=10;
    float avg_parallelTime=0;
    for(int i=0;i<iterations;i++){

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int i, err;

    /* Program/kernel data structures */
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    cl_kernel kernel;

    /* Data and buffers */
    float mat[16], vec[4], result[4];
    float correct[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    cl_mem mat_buff, vec_buff, res_buff;
    size_t work_units_per_kernel;

    cl_mem m_dPingArray;
    cl_mem m_string1;
    cl_mem m_string2;
    cl_mem m_row;
    cl_mem m_column;
    cl_mem m_max_row;
    cl_mem m_max_column;
    cl_mem sub_buffer;
    cl_mem m_test;
    cl_mem m_indicate;
    cl_kernel edit_distance_algo;
    size_t m_N;
    size_t m_N_padded;
    FILE *infile1, *infile2;
    char *a, *b;
    long sizea, sizeb;

    infile1 = fopen("str1.txt", "r");
    infile2 = fopen("str2.txt", "r");

    if (infile1 == NULL)
    {
        printf("File Not Found!\n");
        return -1;
    }
    if (infile2 == NULL)
    {
        printf("File Not Found!\n");
        return -1;
    }

    fseek(infile1, 0L, SEEK_END);
    sizea = ftell(infile1);

    fseek(infile2, 0L, SEEK_END);
    sizeb = ftell(infile2);

    fseek(infile1, 0L, SEEK_SET);
    fseek(infile2, 0L, SEEK_SET);

    a = (char *)calloc(sizea + 1, sizeof(char));
    b = (char *)calloc(sizeb + 1, sizeof(char));

    if (a == NULL)
        return 1;

    fread(a, sizeof(char), sizea, infile1);
    fclose(infile1);

    fread(b, sizeof(char), sizeb, infile2);
    fclose(infile2);

    //char a[7] = "sunday";
    //char b[9] = "saturday";
    //long sizea, sizeb;
    //a[6] = '\0';
    //b[8] = '\0';
    //sizea = 6;
    //sizeb = 8;
    a[sizea] = '\0';
    b[sizeb] = '\0';
    static long int str1len = sizea;
    static long int str2len = sizeb;

    printf("%d %d %d %d\n", str1len, str2len, strlen(a), strlen(b));

    cl_uint input_array[str1len+1][str2len+1];
    //cl_uint** input_array = malloc(sizeof(cl_uint) * (strlen(a)+1));
    clock_t t;

    for (int i = 0; i < str1len + 1; i++)
    {
        input_array[0][i] = i;
    }
    for (int i = 0; i < str2len + 1; i++)
    {
        input_array[i][0] = i;
    }
    //printf("%d %d %c abg\n", strlen(a), strlen(b),a[0]);
    printf("hkdvkh\n");

    //clock_t begin = clock();
    clock_t end;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    {
        perror("Couldn't find any platforms");
        exit(1);
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err < 0)
    {
        perror("Couldn't find any devices");
        exit(1);
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    program_handle = fopen(PROGRAM_FILE, "rb");
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = clCreateProgramWithSource(context, 1,
                                        (const char **)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {

        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    cl_int clError, clError2;
    m_dPingArray = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * (str1len + 1) * (str2len + 1), NULL, &clError2);
    clError = clError2;
    m_string1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char) * str1len, NULL, &clError2);
    clError |= clError2;
    m_string2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char) * str2len, NULL, &clError2);
    clError |= clError2;
    m_row = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * 1, NULL, &clError2);
    clError |= clError2;
    m_column = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * 1, NULL, &clError2);
    clError |= clError2;
    m_max_row = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * sizeof(cl_uint), NULL, &clError2);
    clError |= clError2;
    m_max_column = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * sizeof(cl_uint), NULL, &clError2);
    clError |= clError2;
    m_test = clCreateBuffer(context, CL_MEM_READ_WRITE, 5 * sizeof(cl_uint), NULL, &clError2);
    clError |= clError2;
    m_indicate = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * sizeof(cl_uint), NULL, &clError2);
    cl_mem m_size = clCreateBuffer(context, CL_MEM_READ_WRITE, 1 * sizeof(cl_uint), NULL, &clError2);
    clError |= clError2;

    if (clError < 0)
    {
        printf("pm %d", clError);
        exit(1);
    }

    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0)
    {
        perror("Couldn't create the command queue");
        exit(1);
    }

    edit_distance_algo = clCreateKernel(program, "edit_distance_algo", &clError);
    if (clError < 0)
    {
        printf("qweer %d", clError);
        exit(1);
    }

    size_t globalWorkSize[1];
    size_t localWorkSize[1];

    if (clError < 0)
    {
        printf("op %d", clError);
        exit(1);
    }
    t = clock();
    int max_row = str2len + 1;
    int max_column = str1len + 1;
    int row;
    int column;
    int indicate;
    int test[5];
    int g = 0;
    
    int wr;
    localWorkSize[0] = 256;
    clEnqueueWriteBuffer(queue, m_dPingArray, CL_FALSE, 0, max_row * max_column * sizeof(cl_uint), input_array[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, m_string1, CL_FALSE, 0, str1len * sizeof(char), a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, m_string2, CL_FALSE, 0, str2len * sizeof(char), b, 0, NULL, NULL);
    clError = clSetKernelArg(edit_distance_algo, 1, sizeof(cl_mem), (void *)&m_string1);
    clError |= clSetKernelArg(edit_distance_algo, 2, sizeof(cl_mem), (void *)&m_string2);
    clError |= clSetKernelArg(edit_distance_algo, 5, sizeof(cl_uint), (void *)&max_row);
    clError |= clSetKernelArg(edit_distance_algo, 6, sizeof(cl_uint), (void *)&max_column);

    for (int h = 2; h < max_column + max_row - 1; h++)
    {
        if (h <= str1len)
        {
            row = 0;
            column = h;
            indicate = 0;
            wr = h;
            globalWorkSize[0] = GetGlobalWorkSize(wr, localWorkSize[0]);
            //globalWorkSize[0] = h;
        }
        else
        {
            row = h - max_column + 1;
            column = max_column - 1;
            indicate = 1;
            wr = max_row - 1 - g;
            globalWorkSize[0] = GetGlobalWorkSize(wr, localWorkSize[0]);
            //globalWorkSize[0] = max_row - 1 - g;
            g += 1;
            //printf("%d %d %d %d in\n", row,column,indicate,globalWorkSize[0]);
            //break;
        }
        
        clError = clSetKernelArg(edit_distance_algo, 0, sizeof(cl_mem), (void *)&m_dPingArray);

        clError |= clSetKernelArg(edit_distance_algo, 3, sizeof(cl_uint), (void *)&row);
        clError |= clSetKernelArg(edit_distance_algo, 4, sizeof(cl_uint), (void *)&column);

        //clError |= clSetKernelArg(edit_distance_algo, 7, sizeof(cl_mem), (void*)&m_test);
        clError |= clSetKernelArg(edit_distance_algo, 7, sizeof(cl_uint), (void *)&indicate);
        clError |= clSetKernelArg(edit_distance_algo, 8, sizeof(cl_uint), (void *)&wr);

        //localWorkSize[0] = LocalWorkSize[0];
        if (clError < 0)
        {
            perror("Couldn't enqueue the kernel execution command");
            exit(1);
        }
        clError = clEnqueueNDRangeKernel(queue, edit_distance_algo, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if (clError < 0)
        {
            perror("Couldn't enqueue the kernel execution command");
            exit(1);
        }
        
    }
    //double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
    clError = clEnqueueReadBuffer(queue, m_dPingArray, CL_TRUE, 0, max_row * max_column * sizeof(cl_uint), input_array[0], 0, NULL, NULL);
    if (clError < 0)
    {
        perror("Couldn't enqueue the kernel execution command");
        exit(1);
    }
    end = clock();
    double time_spent = 0.0;
    time_spent = (double)(end - t) / CLOCKS_PER_SEC;

    printf("\n\nEditDist() took %f seconds to execute \n\n", time_spent);
    avg_parallelTime+=time_spent;
    //printf("%d %d \n", strlen(a), strlen(b));
    /*for (int h = 0;h < strlen(b) + 1;h++)
    {
        for (int j = 0;j < strlen(a) + 1;j++)
        {
            printf("%d ", input_array[h][j]);
        }
        printf("\n");
    }*/
    printf("%d", input_array[str2len][str1len]);

    clReleaseMemObject(m_dPingArray);
    clReleaseMemObject(m_string1);
    clReleaseMemObject(m_string2);
    clReleaseMemObject(m_row);
    clReleaseMemObject(m_column);
    clReleaseKernel(edit_distance_algo);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);

    }

    printf("\n\nAverage time :%d",avg_parallelTime/iterations);
}