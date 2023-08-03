CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -O2 -Wall
PROFILE_FLAGS = -pg -no-pie -fno-builtin

all: my_gemm cpu_gemm_openblas
my_gemm: my_gemm.cu
	nvcc my_gemm.cu -pg -g -o my_gemm -lcublasLt -lcurand

cpu_gemm_openblas.o: cpu_gemm_openblas.c
	gcc -I/usr/include/OpenBLAS/include -c -O2 cpu_gemm_openblas.c 

cpu_gemm_openblas: cpu_gemm_openblas.o
	gcc -L/usr/include/OpenBLAS/libopenblas.a -o cpu_gemm_openblas cpu_gemm_openblas.o -lm -lopenblas


cpu_gemm_gsl.o: cpu_gemm_gsl.c
	gcc -Wall -I/usr/local/gsl -c cpu_gemm_gsl.c

cpu_gemm_gsl: cpu_gemm_gsl.o
	gcc -L/usr/local/lib -o cpu_gemm cpu_gemm.o -lgsl -lgslcblas -lm