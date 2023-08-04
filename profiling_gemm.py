import sys
import os
import subprocess
import sqlite3
import pandas as pd
import numpy as np
import time
import glob

MAX_BYTES_GPU = 24576000000

PROFILE_COMMAND_PREF = "ncu --set=full --kernel-name=regex:Kernel --page=details --print-details=all --csv -f --export="

OUTPUT_TRACE_DIR = "/mnt/storage/research/princeton/resource_estimation/gemm/3090/"

PROFILE_DIR = "/mnt/storage/research/princeton/resource_estimation/gemm/3090/"

EXECUTABLE = "./my_gemm"


def build_profile_command(m, k, n):

	output_filename = OUTPUT_TRACE_DIR + str(m) + "_" + str(k) + "_" + str(n)
	profile_command = PROFILE_COMMAND_PREF + output_filename + " " + EXECUTABLE + " " + str(m) + " " + str(k) + " " + str(n) + " > " + output_filename + ".csv"

	return output_filename, profile_command


def main():

	

	os.chdir(PROFILE_DIR)

	csv_files = sorted([f for f in glob.glob("*.csv")])

	exiting_profiled = set()

	for f in csv_files:
		params = f.split(".")[0].split("_")
		m = int(params[0])
		k = int(params[1])
		n = int(params[2])
		exiting_profiled.add((m, k, n))

	dim_list = [1024 * i for i in range(1, 65)]

	for m in dim_list:
		for k in dim_list:
			for n in dim_list:
				mem_usage = 4 * (m * k + k * n + m * n)
				if (mem_usage > MAX_BYTES_GPU - 2000000000):
					continue
				if (m, k, n) in exiting_profiled:
					continue
				output_filename, profile_command = build_profile_command(m, k, n)
				print("Executing command: " + profile_command)
				subprocess.call(profile_command, shell = True)


if __name__ == "__main__":
    main()