# Compiler definitions
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -std=c++11
OPENMP_FLAGS = -fopenmp -DENABLE_OPENMP
MATH_LIB = -lm

# Source files
SRC_CPP = matrix_io.cpp
SRC_SEQUENTIAL = attention_sequential.cpp
SRC_OPENMP = attention_openmp.cpp
SRC_CUDA = attention_cuda.cu
SRC_CUDA_TILE = attention_cuda_tile.cu
SRC_FLASH = attention_flash.cu

# Executables
TARGETS = attention_sequential attention_openmp attention_cuda attention_cuda_tile attention_flash

# Build rules
all: $(TARGETS)

# Build attention_sequential
attention_sequential: $(SRC_SEQUENTIAL) $(SRC_CPP)
	$(CXX) $(SRC_SEQUENTIAL) $(SRC_CPP) -o $@ $(CXXFLAGS)

# Build attention_openmp
attention_openmp: $(SRC_OPENMP) $(SRC_CPP)
	$(CXX) $(OPENMP_FLAGS) $(SRC_OPENMP) $(SRC_CPP) -o $@ $(CXXFLAGS)

# Build attention_cuda
attention_cuda: $(SRC_CUDA) $(SRC_CPP)
	$(NVCC) $(SRC_CUDA) $(SRC_CPP) $(MATH_LIB) -o $@

# Build attention_cuda_tile
attention_cuda_tile: $(SRC_CUDA_TILE) $(SRC_CPP)
	$(NVCC) $(SRC_CUDA_TILE) $(SRC_CPP) $(MATH_LIB) -o $@

# Build attention_flash
attention_flash: $(SRC_FLASH) $(SRC_CPP)
	$(NVCC) $(SRC_FLASH) $(SRC_CPP) $(MATH_LIB) -o $@

# Run rules
run: $(TARGETS)
	@echo "Running toy tests..."
	@echo "num_batch=2 sequence_length=2 embed_dim=2"
	@echo -e "\n*** attention_sequential ***"
	./attention_sequential -b 2 -e 2 -s 2 -t
	@echo -e "\n***attention_openmp ***"
	./attention_openmp -b 2 -s 2 -e 2 -t
	@echo -e "\n***attention_cuda simple ***"
	./attention_cuda -b 2 -s 2 -e 2 -t
	@echo -e "\n***attention_cuda transposed ***"
	./attention_cuda -b 2 -s 2 -e 2 -m t -t
	@echo -e "\n***attention_cuda_tile single ***"
	./attention_cuda_tile -b 2 -s 2 -e 2 -t
	@echo -e "\n***attention_cuda_tile reduced ***"
	./attention_cuda_tile -b 2 -s 2 -e 2 -m r -t
	@echo -e "\n***attention_flash single ***"
	./attention_flash -b 2 -s 2 -e 2 -t
	@echo -e "\n***attention_flash multi ***"
	./attention_flash -b 2 -s 2 -e 2 -m m -t

# Clean up build artifacts
clean:
	rm -f $(TARGETS)

.PHONY: all clean run

# make clean && make && make run > ../output/output_toys.txt