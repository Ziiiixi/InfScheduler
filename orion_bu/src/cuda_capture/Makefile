# Compiler and flags
CC=g++
CUDAINCLUDE=/usr/local/cuda-11.8/include/
CUDALIB=/usr/local/cuda-11.8/lib64

# Debug flag
DEBUG_FLAGS=

# Uncomment the following line to enable debug mode
# DEBUG_FLAGS=-DDEBUG

# Target to build the shared library
libinttemp.so: utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp
	$(CC) $(DEBUG_FLAGS) -O3 -std=c++17 -fPIC -shared utils_interc.cpp intercept_cudnn.cpp intercept_cublas.cpp intercept_temp.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lcublas -lpthread -o libinttemp.so

all: libinttemp.so

clean:
	rm -rf *.o libinttemp.so
