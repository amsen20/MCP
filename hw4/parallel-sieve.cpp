#define _CRT_SECURE_NO_WARNINGS

#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

const int FAST_START = 100, IT_NUM = 10;

int main(int argc, char **argv) {
    int N = atoi(argv[1]);
    bool *mark = new bool[N];

    int threads_num = atoi(argv[2]);
    omp_set_num_threads(threads_num);
    
    double all_elapsedtime = 0;

    for(int _=0 ; _<IT_NUM ; _++) {
        memset(mark, 0, sizeof mark);
        
        double starttime = omp_get_wtime();

        int block = min(int(log(N) * threads_num), N);
        int ls=2;
        
        #pragma omp parallel
        #pragma omp single nowait
        {
            for(int i=2 ; i*i<=block ; i++) {
                ls = i;
                if(!mark[i])
                    for(int id=0 ; id<threads_num-1 ; id++)
                    #pragma omp task firstprivate(i, id)
                    {
                        for(int j=i*i + id*i ; j<=N ; j += (threads_num-1) * i)
                            mark[j] = true;
                    }
            }
        }

        #pragma omp parallel
        #pragma omp single nowait
        {
            for(int st=ls+1 ; st*st<=N ; st+=block) {
                for(int i=st ; i*i<=N && i<st+block ; i++)
                    if(!mark[i]) {
                        #pragma omp task firstprivate(st)
                        {
                        for(int j=i*i ; j<=N ; j += i)
                            mark[j] = true;
                        }
                    }
                #pragma omp taskwait
            }
        }
        
        double	elapsedtime = omp_get_wtime() - starttime;
        all_elapsedtime += elapsedtime;
    }

    int primes_count = 0;

    for(int i = 2; i <= N; ++i){
		if(mark[i] == 0){
			primes_count ++;
		}
	}
	
	cout << primes_count << " Primes in " << all_elapsedtime/IT_NUM << " Second(s)." << std::endl;

    free(mark);
    return 0;
}

/*
    parallel = {
        100000000: 0.339957,
        1000000000: 4.56043,
        10000000000: 7.04847
    }
*/