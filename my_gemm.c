#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cublasLt.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <stdint.h>


int main(int argc, char * argv[]){

	if (argc != 4){
		fprintf(stderr, "Wrong number of args\n");
		exit(1);
	}

	int m = atoi(argv[1]);
	int k = atoi(argv[2]);
	int n = atoi(argv[3]);

	if (m <= 0 || k <= 0 || n <= 0){
		fprintf(stderr, "Bad dimensions\n");
		exit(1);
	}

	// using cuRand to populate matrices
	curandGenerator_t gen;
	curandStatus_t curand_status;
	curand_status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curand_status = curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

	// Init Matrices
	float * A, *B, *C;
	cudaMalloc(&A, m * k * sizeof(float));
	cudaMalloc(&B, k * n * sizeof(float));
	cudaMalloc(&C, m * n * sizeof(float));

	// choosing arbitrary stddev that might resemeble init weights of Neural Network
	curand_status = curandGenerateNormal(gen, A, (size_t) (m * k), 0, sqrtf(1.0 / 20.0));
	curand_status = curandGenerateNormal(gen, B, (size_t) (k * n), 0, sqrtf(1.0 / 20.0));

	curand_status = curandDestroyGenerator(gen);

	// PERFORM MATMUL

	cublasStatus_t status;
	cublasLtHandle_t handle;
	status = cublasLtCreate(&handle);

	cublasLtMatrixLayout_t Adesc;
	cublasLtMatrixLayout_t Bdesc;
	cublasLtMatrixLayout_t Cdesc;

	status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, m, k, 0);
	status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k, n, 0);
	status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, 0);


	cublasLtMatmulDesc_t matmulDesc;

	status = cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F_FAST_16BF, CUDA_R_32F);


	cublasLtMatmulPreference_t pref;

	// ALLOW 1 GB of workspace mem...
	const size_t workspaceBytes = 1000000000;
	status = cublasLtMatmulPreferenceCreate(&pref);
	status = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceBytes, sizeof(workspaceBytes));

	int algoCount = 5;
	int retAlgoCount;

	cublasLtMatmulHeuristicResult_t heuristicResultsArray[algoCount];


	status = cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Adesc, Bdesc, Cdesc, Cdesc, pref, algoCount, heuristicResultsArray, &retAlgoCount);

	cublasLtMatmulAlgo_t algo = heuristicResultsArray[0].algo;

	void * workspace;
	cudaMalloc(&workspace, workspaceBytes);

	float alpha = 1, beta = 0;

	status = cublasLtMatmul(handle,
							matmulDesc,
							&alpha,
							A,
							Adesc,
							B,
							Bdesc,
							&beta,
							C,
							Cdesc,
							C,
							Cdesc,
							&algo,
							workspace,
							workspaceBytes,
							0);


	// FREE workspace


	cudaFree(workspace);


	// FREE cuBlasLt Structs 

	status = cublasLtMatmulPreferenceDestroy(pref);
	status = cublasLtMatmulDescDestroy(matmulDesc);

	status = cublasLtMatrixLayoutDestroy(Adesc);
	status = cublasLtMatrixLayoutDestroy(Bdesc);
	status = cublasLtMatrixLayoutDestroy(Cdesc);

	status = cublasLtDestroy(handle);


	// FREE MATRICES

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

}