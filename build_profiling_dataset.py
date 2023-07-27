import os
import glob
import pandas as pd
import pickle

PROFILE_DIR = "/mnt/storage/research/princeton/resource_estimation/gemm/3090/"

os.chdir(PROFILE_DIR)

csv_files = sorted([f for f in glob.glob("*.csv")])

main_df = None

for f in csv_files:

	params = f.split(".")[0].split("_")

	m = params[0]
	k = params[1]
	n = params[2]

	try:
		df = pd.read_csv(f, skiprows=4)
	

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
		print(e)
		print()
		continue


main_df = main_df.sort_values(by=["M", "K", "N"], ignore_index=True)

with open(PROFILE_DIR + "profiling_df.pickle", "wb") as out_file:
	pickle.dump(main_df, out_file)
