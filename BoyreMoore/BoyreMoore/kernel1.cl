__kernel void search(__global char *inputText, __global char *searchPattern,
                     __global int *searchIndex, __global int *result,
                     __global int *goodSymTab, __global int *badSymTab,
                     int sublength) {
  int id = get_global_id(0);
  int occurence = 0;
  int shift = 0;
  result[id] = 0;
  int k;
  int num;

  int len = sublength;

  int dlen = searchIndex[id * 2 + 1];
  int i = searchIndex[id * 2] + len - 1;

  while (i <= dlen) {
    k = 0;
    while (k <= len - 1 && inputText[i - k] == searchPattern[len - 1 - k]) {
      k = k + 1;
    }

    if (k == len) {
      printf("Found by %d at : %d\n", id, (i - (len - 1)));
      i = i + 1;
      result[id] = ++occurence;
      continue;
    }
    num = (int)inputText[i];

    if (badSymTab[num] - k) > 1){
      int d1 = badSymTab[num] - k
    }
    else {
      d1 = 1;
    }

    int d2 = goodSymTab[k];
    if (k == 0) {
      shift = d1;
    } else if (k > 0) {
      if (d1 > d2) {
        shift = d1;
      } else {
        shift = d2;
      }
    }
    i = i + shift;
  }
}