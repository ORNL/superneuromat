#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>


int test_openmp(int size)
{
	// Data
	double A[size], B[size], C[size];

	// Initialize array
	time_t start, end;

	start = clock();

	// #pragma omp parallel
	// {
	// 	printf("Number of openmp threads: %d\n", omp_get_num_threads());
	// }

	#pragma omp parallel for num_threads(64)
	for (int i = 0; i < size; i++)
	{
		A[i] = ((double)rand() / RAND_MAX);
		B[i] = ((double)rand() / RAND_MAX) * 10.0;
		C[i] = (log(A[i] + B[i]) * sin(A[i] + B[i])) * (log(A[i] - B[i]) * sin(A[i] - B[i]));
	}

	end = clock();


	// Print array
	// printf("Array A: [ ");
	// for (int i = 0; i < size; i++)
	// {
	// 	printf("%d ", A[i]);
	// }
	// printf("]\n");

	printf("Time taken: %.4f us\n", 1000000*((double)(end - start))/CLOCKS_PER_SEC);

	return 0;
}



int main()
{
	test_openmp(300000);

	return 0;
}
