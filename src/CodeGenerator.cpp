

#include <iostream>

int main() {
	for( int i = 32 ; i <= 1024 ; i += 32 ) {
		std::cout << "GLOBALS(" << i << ")" << std::endl;
	}
	return 0;
}