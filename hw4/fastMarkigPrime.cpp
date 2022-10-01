/*
*				In His Exalted Name
*	Title:	Fast Marking Technique
*	Author: Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	Date:	28/12/2013
*/

#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>

const int IT_NUM = 10;

void usage(void){
	std::cout << "sieve <max_number>" << std::endl;
	std::cout << "<max_number> range between 2 and N." << std::endl;
}

int main(int argc, char **argv){
	if(argc != 2){
		std::cout << "Invalid number of arguments!" << std::endl;
		usage();
		return 0;
	}
	int range_max = atoi(argv[1]);
	if (range_max < 2){
		std::cout << "<max_number> Must be greater than or equal to 2." << std::endl;
		usage();
		return 0;
	}
	// block of data initialized to zero
	char * marked = (char *) calloc(sizeof (char),range_max);

	if(marked == 0){
		std::cout << "Cannot allocate enough memory." << std::endl;
		exit(1);
	}

	// Global count
	int count = 0;

	int all = 0;
	for(int _=0 ; _<IT_NUM ; _++) {
		
		memset(marked, 0, sizeof marked);
		// Global k
		int k = 2;

		// Global index
		int prime_index = 0;

		int low_value = 2;

		int first_index = 0;
		
		clock_t t = clock();
		
		do{
			if (k > low_value){
				first_index = k - low_value + k;
			}else if(k*k > low_value){
				first_index = k * k - low_value;
			}else if(low_value % k == 0){
				first_index = 0;
			}else{
				first_index = k - (low_value % k);
			}

			for(int i = first_index; i < range_max; i += k){
				marked[i] = 1;
			}
			while(marked[++prime_index]);
			k = prime_index + 2;
		}while(k*k <= range_max);
		
		
		all += clock() - t;
	}
	
	for(int i = 0; i < range_max; ++i){
		if(marked[i] == 0){
			++count;
		}
	}
	std::cerr << all << "\n";
	std::cout << count << " Primes in " << ((float)all)/CLOCKS_PER_SEC/IT_NUM << " Second(s)." << std::endl;
	
	free(marked);
	marked = NULL;

	std::cout << count << "primes found between 2 and " << range_max << std::endl;
	return 0;
}

/*
	serial = {
		100000000: 0.852057,
		1000000000: 10.284,
		10000000000: 14.6295
	}
*/