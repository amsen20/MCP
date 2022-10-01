
#include <stdio.h>
#include <math.h>
#include <omp.h>

const int VERYBIG = 50000;
// **********************************************************************
int main(void)
{
	#ifndef _OPENMP
		printf("OpenMp is not supported\n");
		getchar();
		return 0;
	#else
		printf("OpenMp is good :)\n");
	#endif

	int i;
	double starttime, elapsedtime;
	// -----------------------------------------------------------------------
	// Output a start message
	printf("Serial Timings for %d iterations\n\n", VERYBIG);
	// repeat experiment several times
	for (i = 0; i<10; i++)
	{
		long int j, k;
		double sumx, sumy;
		// get starting time56 x CHAPTER 3 PARALLEL STUDIO XE FOR THE IMPATIENT
		starttime = omp_get_wtime();
		// reset check sum & running total
		int sum[VERYBIG] = {};
		double total[VERYBIG] = {};
		#pragma omp parallel for private(k, sumx, sumy)
		// Work Loop, do some work by looping VERYBIG times
			for (j = 0; j<VERYBIG; j++)
			{
				// increment check sum
				sum[j] += 1;
				// Calculate first arithmetic series
				sumx = 0.0;
				for (k = 0; k<j; k++)
					sumx = sumx + (double)k;
				// Calculate second arithmetic series
				sumy = 0.0;
				for (k = j; k>0; k--)
					sumy = sumy + (double)k;
				
				if (sumx > 0.0) total[j] = 1.0 / sqrt(sumx);
				if (sumy > 0.0) total[j] += 1.0 / sqrt(sumy);
			}
		// get ending time and use it to determine elapsed time
		elapsedtime = omp_get_wtime() - starttime;
		// report elapsed time
		long int _sum = 0;
		double _total = 0.0;
		for(int i=0 ; i<VERYBIG ; i++)
			_sum += sum[i];
		for(int i=0 ; i<VERYBIG ; i++)
			_total += total[i];
		printf("Time Elapsed: %f Secs, Total = %lf, Check Sum = %ld\n",
			elapsedtime, _total, _sum);
	}
	// return integer as required by function header
	getchar();
	return 0;
}
