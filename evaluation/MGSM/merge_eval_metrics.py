import os

import pandas as pd


def merge_excel_files():
	# Define the possible modes
	icl_modes = ["native", "english", "multilingual"]  # Add more if needed
	# cot_modes = ["direct", "native", "english"]  # Add more if needed
	cot_modes = ["english"]  # Add more if needed

	# Initialize an empty list to store all dataframes
	all_dfs = []

	# Iterate through all combinations of icl_mode and cot_mode
	parent_dir = "deprecated/20240713_deterministic_sampling/mgsm_eval/llama2-7b-chat"
	for icl_mode in icl_modes:
		for cot_mode in cot_modes:
			folder_name = f"icl-{icl_mode}_cot-{cot_mode}"
			folder_name = os.path.join(parent_dir, folder_name)
			file_path = os.path.join(folder_name, "mgsm_evaluation_metrics.xlsx")

			# Check if the file exists
			if os.path.exists(file_path):
				df = pd.read_excel(file_path, index_col='Language')

				# Add columns for icl_mode and cot_mode
				df['ICL_Mode'] = icl_mode
				df['CoT_Mode'] = cot_mode

				# Append to the list of dataframes
				all_dfs.append(df)
			else:
				print(f"File not found: {file_path}")

	# Concatenate all dataframes
	if all_dfs:
		merged_df = pd.concat(all_dfs)

		# Reset index to make 'Language' a column again
		merged_df = merged_df.reset_index()

		# Reorder columns
		columns_order = ['ICL_Mode', 'CoT_Mode', 'Language', 'Accuracy', 'Precision', 'Recall', 'F1']
		merged_df = merged_df[columns_order]

		# Save the merged dataframe to a new Excel file
		output_file = "merged_mgsm_evaluation_metrics.xlsx"
		output_file = os.path.join(parent_dir, output_file)
		merged_df.to_excel(output_file, index=False)
		print(f"Merged results saved to {output_file}")
	else:
		print("No files were found to merge.")


# Run the function
merge_excel_files()
