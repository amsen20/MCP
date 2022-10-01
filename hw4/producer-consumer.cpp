#define _CRT_SECURE_NO_WARNINGS

#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

const int BUFF_SIZE = 2048;

int st, en, done, processed_files_count;
bool buff_state[BUFF_SIZE];
/*
 ---done+++[start****en)---
 -: empty element
 +: elements that are being processed
 *: elements filled with lines and need to be processed
*/
omp_lock_t st_lock, en_lock;

string buffer[BUFF_SIZE];

void produce(const string &file_path) {
	ifstream fin(file_path);
	for(string line ; getline(fin, line) ; ) {
		omp_set_lock(&en_lock); // lock the end pointer
		
		int nen = (en + 1) % BUFF_SIZE;
		// check wether is buffer full or not
		while(1) {
			#pragma omp flush(done)
			if(done != nen)
				break;
		}
		
		int ind = en;
		en = nen; // just move the end pointer and store it to ind
		#pragma omp flush(en)
		
		omp_unset_lock(&en_lock); // release the end pointer

		buffer[ind] = line; // load the new line to the buffer
		#pragma omp flush

		buff_state[ind] = true; // set state of ind th element of buffer to ready
		#pragma omp flush

	}
}

void consume() {
	while(1) {

		// check is buffer empty or not
		omp_set_lock(&st_lock); // lock the start pointer
		
		#pragma omp flush(en)
		if(st == en) {
			omp_unset_lock(&st_lock);
			continue;
		}
		
		int ind = st;
		st = (st + 1) % BUFF_SIZE; // just move the start pointer and store it to ind
		#pragma omp flush(en)

		omp_unset_lock(&st_lock); // release the start pointer

		// wait until the line is fully copied to the buffer element
		while(1) {
			#pragma omp flush
			if (buff_state[ind] > 0)
				break;
		}
		string line = buffer[ind];
		buff_state[ind] = false;
		#pragma omp flush
		
		// wait until the done pointer is on ind
		while (1) {
			#pragma omp flush (done)
			if(done == ind)
				break;
		}

		done = (done + 1) % BUFF_SIZE; // free the buffer element located in done
		#pragma omp flush(done)

		// process the line
		string token;
		for(auto ch: line)
			if(ch == ' ') {
				#pragma omp critical // in order to display stdout well, you can comment it.
				cout << "Found token: " << token << endl;
				token = "";
			} else {
				token.push_back(ch);
			}
		
		if(!token.empty()) {
			#pragma omp critical // in order to display stdout well, you can comment it.
			cout << "Found token: " << token << endl;
		}
	}
}

int main() {
	cout << "Enter number of files: " << "\n";
	int file_count;
	cin >> file_count;

	vector<string> file_paths;
	for(int i=0 ; i<file_count ; i++) {
		cout << "Enter " << i+1 << "th file's path: " << "\n";
		string path;
		cin >> path;
		file_paths.push_back(path);
	}
	cout << endl;

	int max_threads = omp_get_max_threads();
	omp_init_lock(&st_lock); // the start pointer's lock
	omp_init_lock(&en_lock); // the end pointer's lock

	assert(file_paths.size() < max_threads);

	int files_done = 0;
	#pragma omp parallel num_threads(max_threads)
	{
		int id = omp_get_thread_num();

		if(id < file_count) {
			produce(file_paths[id]);
			
			#pragma omp atomic update
			files_done ++;
			if(files_done == file_count) {
				while(1) {
					#pragma omp flush
					if(done == en)
						break;
				}
				exit(0);
			}

		} else
			consume();
	}
}
