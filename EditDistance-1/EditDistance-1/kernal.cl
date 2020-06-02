#define MAX_LOCAL_SIZE 16

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void edit_distance_algo(__global uint *inArray, __global char *a,
                                 __global char *b, const uint row,
                                 const uint column, const uint max_row,
                                 const uint max_column,
                                 const uint indicate, const uint size)
{
  //get the global unique id of thread executing this kernel.
  int gid;
  gid = get_global_id(0);
  int max_col;
  //initialise the arguments to a private memory variable.
  max_col = max_column;
  
  //allow only valid thread to enter and update the result.
  if (gid < size && (gid > 0 || indicate == 1))
  {
    uint r;
    //get index of current row to be operated on.
    r = (row + gid);
    uint c;
    //get index of current column to be operated on.
    c = column - gid;
    uint current_index;
    //Converting 2d array index into 1d array index.
    current_index = r * max_col + c;
    uint diagonal;
    //getting index of diagonal top left element relative to current working element in 1D representation.
    diagonal = (r - 1) * max_col + c - 1;
    //for the current working cell, get the corresponding character of string1 and string2 and check if both are equal.
    if (b[r - 1] == a[c - 1])
    {
      //take value of diagonal top left element.
      inArray[current_index] = inArray[diagonal];
    }
    
    else
    {
      //if both character are different ,take minimum value among the 3 cells.
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