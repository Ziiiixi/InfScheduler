# Compiler and flags
CC=g++
NVCC=/usr/local/cuda/bin/nvcc
CFLAGS=-O3 -std=c++17
CUDAINCLUDE=/usr/local/cuda-11.8/include/
CUDALIB=/usr/local/cuda-11.8/lib64
SMCTRL_CFLAGS=-fPIC
SMCTRL_LDFLAGS=-lcuda -I/usr/local/cuda/include -ldl

# Debug flag
DEBUG_FLAGS=

# Uncomment the following line to enable debug mode
# DEBUG_FLAGS=-DDEBUG

# Object files
OBJ_FILES=utils_sched.o scheduler_eval.o libsmctrl.o

# Targets
all: scheduler_eval.so

libsmctrl.o: libsmctrl.cpp libsmctrl.h
	$(CC) $(SMCTRL_CFLAGS) $(DEBUG_FLAGS) -c libsmctrl.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcuda -o libsmctrl.o

utils_sched.o: utils_sched.cpp
	$(NVCC) $(CFLAGS) $(DEBUG_FLAGS) -Xcompiler -fPIC -c utils_sched.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lcublas -lpthread

scheduler_eval.o: scheduler_eval.cpp
	$(NVCC) $(CFLAGS) $(DEBUG_FLAGS) -Xcompiler -fPIC -c scheduler_eval.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lcublas -lpthread

scheduler_eval.so: $(OBJ_FILES)
	$(CC) $(CFLAGS) $(DEBUG_FLAGS) -fPIC -shared $(OBJ_FILES) -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lcuda -lcublas -lpthread -o scheduler_eval.so

clean:
	rm -rf *.o *.so
