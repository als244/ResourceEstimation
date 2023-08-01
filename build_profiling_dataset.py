import os
import glob
import pandas as pd
import pickle

PROFILE_DIR = "/mnt/storage/research/princeton/resource_estimation/gemm/3090/"

os.chdir(PROFILE_DIR)

csv_files = sorted([f for f in glob.glob("*.csv")])

main_df = None

error_files = []

for f in csv_files:

	params = f.split(".")[0].split("_")

	m = int(params[0])
	k = int(params[1])
	n = int(params[2])

	try:
		df = pd.read_csv(f, skiprows=3)
	

		## assume name kernel name, block size, grid size for all metrics
		kern_name = df.loc[0, "Kernel Name"]
		block_size = df.loc[0, "Block Size"]
		grid_size = df.loc[0, "Grid Size"]
	

		df_metrics = df[["Metric Name", "Metric Unit", "Metric Value"]]

		df_metrics = df_metrics[df_metrics["Metric Value"].notna()]

		metric_names_with_unit = df_metrics["Metric Name"] + " (" + df_metrics["Metric Unit"] + ")"

		new_cols = ["M", "K", "N", "Kernel Name", "Block Size", "Grid Size"] + list(metric_names_with_unit)

		new_vals = [m, k, n, kern_name, block_size, grid_size] + list(df_metrics["Metric Value"])

		if main_df is not None:
			main_df.loc[len(main_df.index)] = new_vals

		else:
			main_df = pd.DataFrame([new_vals], columns=new_cols)

	except Exception as e:
		print("EXCEPTION for file: " + f)
		error_files.append(f)
		print(e)
		print()
		continue


main_df = main_df.sort_values(by=["M", "N", "K"], ignore_index=True)

main_df = main_df.loc[:,~main_df.columns.duplicated()].copy()
main_df = main_df.loc[:, main_df.columns.notna()]

### important columns for now, if we want to cut down on  memory of dataframe could only save these...

# key_cols = ['M', 'K', 'N', 'Duration (nsecond)', 'Compute (SM) Throughput (%)', 'Memory Throughput (%)', 'Memory Throughput (byte/second)', 'DRAM Throughput (%)', 'Mem Busy (%)', 'Max Bandwidth (%)', 'Mem Pipes Busy (%)', 'Theoretical Occupancy (%)', 'Achieved Occupancy (%)', 
#             'Elapsed Cycles (cycle)', 'SM Active Cycles (cycle)', 'SM: Pipe Tensor Cycles Active (%)',
#             'Executed Ipc Active (inst/cycle)', 'Executed Ipc Elapsed (inst/cycle)', 'Issue Slots Busy (%)', 'Issued Ipc Active (inst/cycle)', 'SM Busy (%)', 'Tensor (All) (%)', 'ALU (%)', 'FMA (%)', 'FMA (FP16) (%)', 'FP64 (%)', 'FP64 (DMMA) (%)',  
#             'L1/TEX Cache Throughput (%)', 'L2 Cache Throughput (%)', 'L1/TEX Hit Rate (%)', 'L2 Hit Rate (%)',
#             'Active Warps Per Scheduler (warp)', 'Eligible Warps Per Scheduler (warp)', 'Theoretical Warps Per Scheduler (warp)', 'GPU Maximum Warps Per Scheduler (warp)', 'Active Warps Per Scheduler (warp)', 'Eligible Warps Per Scheduler (warp)', 'Warp Cycles Per Issued Instruction (cycle)', 'Warp Cycles Per Executed Instruction (cycle)',
#             'Avg. Executed Instructions Per Scheduler (inst)', 'Executed Instructions (inst)', 'Avg. Issued Instructions Per Scheduler (inst)', 'Issued Instructions (inst)', 'Registers Per Thread (register/thread)', 'Shared Memory Per Block (byte/block)', 'Shared Memory Configuration Size (byte)', 'Driver Shared Memory Per Block (byte/block)', 'Dynamic Shared Memory Per Block (byte/block)', 'Static Shared Memory Per Block (byte/block)', 'Threads (thread)', 'Avg. Threads Executed (thread)', 'Avg. Predicated-On Threads Executed (thread)',
#             'Theoretical Active Warps per SM (warp)', 'Achieved Active Warps Per SM (warp)',
#             'L1 Wavefronts Shared Excessive (byte)', 'L2 Theoretical Sectors Global Excessive (byte)', 'Branch Instructions Ratio (%)',
#             'SM Frequency (cycle/second)', 'DRAM Frequency (cycle/second)', 
#             'Kernel Name', 'Block Size', 'Grid Size']

with open(PROFILE_DIR + "profiling_df.pickle", "wb") as out_file:
	pickle.dump(main_df, out_file)

with open(PROFILE_DIR + "error_files_list.pickle", "wb") as out_file:
	pickle.dump(error_files, out_file)
