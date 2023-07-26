my_gemm: my_gemm.cu
	nvcc my_gemm.cu -pg -g -o my_gemm -lcublasLt -lcurand