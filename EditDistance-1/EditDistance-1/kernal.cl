#define MAX_LOCAL_SIZE 16

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void edit_distance_algo(__global uint *inArray, __global char *a,
                                 __global char *b, const uint row,
                                 const uint column, const uint max_row,
                                 const uint max_column,
                                 const uint indicate, const uint size)
{
  int gid;
  gid = get_global_id(0);
  int max_col;
  max_col = max_column;
  if (gid < size && (gid > 0 || indicate == 1))
  {
    uint r;
    r = (row + gid);
    uint c;
    c = column - gid;
    uint current_index;
    current_index = r * max_col + c;
    uint diagonal;
    diagonal = (r - 1) * max_col + c - 1;
    if (b[r - 1] == a[c - 1])
    {
      inArray[current_index] = inArray[diagonal];
    }
    else
    {
      uint left;
      left = current_index - 1;
      uint right;
      right = diagonal + 1;
      uint mi;
      mi = inArray[diagonal];
      if (mi > inArray[left])
        mi = inArray[left];
      if (mi > inArray[right])
        mi = inArray[right];
      inArray[current_index] = mi + 1;
    }
  }
}