__kernel void search(__global char *A,__global char *B,__global int *se,__global int *ans,__global int *gstable,__global int *bstable,int  sublength)
{
  int id=get_global_id(0);
    int shift=0;
  int occ=0;
  ans[id]=0;
  int k,num;
 
  int len=sublength; 
 
  int dlen=se[id*2+1];
  int i=se[id*2]+len-1;
  
  while(i<=dlen)
  { k=0;
    while(k<=len-1 && A[i-k]==B[len-1-k])
     k=k+1;
    
    if(k==len) {
      printf("\nFound by %d at : %d",id,(i-(len-1))); i=i+1; ans[id]=++occ; continue;  
      }


    num = (int)A[i];
    int d1=(bstable[num]-k)> 1 ? (bstable[num]-k) : 1;
    int d2=gstable[k];
    if(k==0) {shift=d1;}
    else if(k>0) shift=d1>d2? d1: d2;
    else {   }
    i=i+shift;
  }
 
}