// A Dynamic Programming based C++ program to find minimum 
// number operations to convert str1 to str2 
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include<time.h>
#include<string.h>

#pragma warning(disable: 4996)
// Utility function to find the minimum of three numbers 
int mi(int x, int y, int z)
{
	int mini = x;
	if (mini > y)mini = y;
	if (mini > z)mini = z;
	return mini;
}

int editDistDP(char* str1, char* str2, int m, int n)
{
	// Create a table to store results of subproblems 
	//int dp[m + 1][n + 1]; 

	int** dp = (int**)malloc((m + 1) * sizeof(int*));
	for (int i = 0; i < m + 1; i++)
		dp[i] = (int*)malloc((n + 1) * sizeof(int));

	//printf("hello\n");
	//int *dp = (int*)malloc(sizeof(int)*(m+1)*(n+1));

	// Fill d[][] in bottom up manner 
	for (int i = 0; i <= m; i++) {
		for (int j = 0; j <= n; j++) {
			// If first string is empty, only option is to 
			// insert all characters of second string 
			if (i == 0)
				dp[i][j] = j; // Min. operations = j 

			// If second string is empty, only option is to 
			// remove all characters of second string 
			else if (j == 0)
				dp[i][j] = i; // Min. operations = i 

			// If last characters are same, ignore last char 
			// and recur for remaining string 
			else if (str1[i - 1] == str2[j - 1])
				dp[i][j] = dp[i - 1][j - 1];

			// If the last character is different, consider all 
			// possibilities and find the minimum 
			else
				dp[i][j] = 1 + mi(dp[i][j - 1], // Insert 
					dp[i - 1][j], // Remove 
					dp[i - 1][j - 1]); // Replace 
		}
	}

	return dp[m][n];
}

// Driver program 
int main()
{
	// your code goes here 
	// string str1 = "sundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturday"; 
	// string str2 = "saturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysaturdaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysundaysunday";

	FILE* infile1, * infile2;
	char* a, * b;
	long sizea, sizeb;

	infile1 = fopen("str1.txt", "r");
	infile2 = fopen("str2.txt", "r");

	if (infile1 == NULL) {
		printf("File Not Found!\n");
		return -1;
	}
	if (infile2 == NULL) {
		printf("File Not Found!\n");
		return -1;
	}

	fseek(infile1, 0L, SEEK_END);
	sizea = ftell(infile1);

	fseek(infile2, 0L, SEEK_END);
	sizeb = ftell(infile2);

	fseek(infile1, 0L, SEEK_SET);
	fseek(infile2, 0L, SEEK_SET);

	a = (char*)calloc(sizea + 1, sizeof(char));	
	b = (char*)calloc(sizeb + 1, sizeof(char));

	if (a == NULL)
		return 1;


	fread(a, sizeof(char), sizea, infile1);
	fclose(infile1);

	fread(b, sizeof(char), sizeb, infile2);
	fclose(infile2);

	a[sizea] = '\0';
	b[sizeb] = '\0';
	long int str1len = strlen(a);
	long int str2len = strlen(b);

	printf("%d %d %d %d\n", str1len, str2len, strlen(a), strlen(b));



	// cout<<str1;

	//auto start = high_resolution_clock::now();
	//ios_base::sync_with_stdio(false);
	clock_t begin = clock();

	printf("%d \n", editDistDP(a, b, str1len, str2len));

	clock_t end = clock();
	double  time_spent = 0.0;
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("\n\nEditDist() took %f seconds to execute \n\n", time_spent);


	return 0;
}
