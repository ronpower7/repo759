#include <iostream>
#include <cstdlib>

int main (int argc , char* argv[]) {

	int N = std::atoi(argv[1]);

	std::cout << "The value of N from command line is: " << N << "\n";

	for (int i = 0 ; i <= N ; ++i) {
		printf("%d ",i);
        }
        printf("\n");
	for (int j = N ; j>=0 ; --j) {
		std::cout << j << " " ;
        }
        printf("\n");
       return 0 ;
}


