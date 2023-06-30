import os
import shutil
import time
import pandas as pd
from typing import List
import tempfile

def gather_csv_files(folder_path: str, file_prefix: str) -> List[str]:
    return [f"{folder_path}/{file_prefix}_{str(i)}.csv" for i in range(0,22) 
            if os.path.isfile(f"{folder_path}/{file_prefix}_{str(i)}.csv")]

def read_csv_files(file_list: List[str]) -> pd.DataFrame:
    df_list = [pd.read_csv(csv_file, low_memory=False, index_col=False) for csv_file in file_list]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.round(5)
    combined_df.sort_values(['Total Frame','Time reduce INF time'], ascending=True, inplace=True)
    combined_df.drop_duplicates(subset='filename', keep='first', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df

def enrich_df(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    df['benchmark'] = df['filename'].apply(lambda x: "hwmcc20" if "hwmcc20" in x else benchmark)
    df['filename'] = df['filename'].apply(lambda x: x.split('/')[-1].replace('.aag', ''))
    return df

def calc_difference(df: pd.DataFrame, col: str, new_col: str) -> pd.DataFrame:
    df[new_col] = df[col].apply(lambda x: str((x-df[col+' (without NN)'])/df[col+' (without NN)']*100).round(1)+'%' if x != df[col+' (without NN)'] else '0%')
    return df

def process_df(folder_path: str, file_prefix: str, benchmark: str):
    csv_file_list = gather_csv_files(folder_path, file_prefix)
    df = read_csv_files(csv_file_list)
    df = enrich_df(df, benchmark)
    return df

folder_path = "../log"
file_prefix_with_NN = "small_subset_experiment_with_NN_subset"
file_prefix_without_NN = "small_subset_without_NN_subset"

df_with_NN = process_df(folder_path, file_prefix_with_NN, "UNKNOWN")
df_without_NN = process_df(folder_path, file_prefix_without_NN, "hwmcc07")

df_with_NN.rename(columns={'Time reduce INF time' : 'Time consuming (without INF time)','Number of clauses': 'clauses'},inplace=True)
df_without_NN.rename(columns={'Total Frame':'Total Frame (without NN)', 'Number of clauses': 'clauses (without NN)', 'Time Consuming':'Time consuming (without NN)'}, inplace=True)

final_result = pd.merge(df_without_NN, df_with_NN, on='filename', how='left')
final_result = final_result.reindex(sorted(final_result.columns,reverse=True), axis=1)
final_result = calc_difference(final_result, 'Time consuming', 'Time reduce')
final_result = calc_difference(final_result, 'clauses', 'Clauses changed')
final_result = calc_difference(final_result, 'Total Frame', 'Frames changed')

# Change the type of coloumn of Total Frame to int
final_result_latex['Total Frame'] = final_result_latex['Total Frame'].astype(int)

# print final_result_latex as latex
final_result_latex = final_result_latex.round(1)
print(final_result_latex.to_latex(index=False))

