import imp
import shutil
from typing import final
import pandas as pd
from IPython.display import display
import os.path
import tempfile
import time
from datetime import datetime
import numpy as np

# Get .csv file list in log folder
csv_file_lst_with_NN = [f"../log/small_subset_experiment_with_NN_subset_{str(i)}.csv" for i in range(0,22) if os.path.isfile(f"../log/small_subset_experiment_with_NN_subset_{str(i)}.csv")]
csv_file_lst_without_NN = [f"../log/small_subset_without_NN_subset_{str(i)}.csv" for i in range(0,22) if os.path.isfile(f"../log/small_subset_without_NN_subset_{str(i)}.csv")]

# walk through the csv_file_with_NN and csv_file_without_NN in ../log/
csv_data_with_NN_frame_lst = [pd.read_csv(csv_with_NN, low_memory = False, index_col=False) for csv_with_NN in csv_file_lst_with_NN[:]]
csv_data_without_NN_frame_lst = [pd.read_csv(csv_without_NN, low_memory = False, index_col=False) for csv_without_NN in csv_file_lst_without_NN[:]]

# concat all the dataframe with ignore index (which is important!)
csv_df_with_NN = pd.concat(csv_data_with_NN_frame_lst,ignore_index=True)
csv_df_without_NN = pd.concat(csv_data_without_NN_frame_lst,ignore_index=True)

# add a new column to the dataframe to indicate where the data come from
csv_df_with_NN['benchmark'] = "UNKNOWN"

# add a new column to the dataframe to calculate how much time has been reduced
csv_df_with_NN['Time reduce'] = 0

# add a new column to the dataframe to calculate how many clauses has been reduced
csv_df_with_NN['Clauses changed'] = 0

# add a new column to the dataframe to calculate how many frames has been reduced
csv_df_with_NN['Frames changed'] = 0

# update the value in the first row, make file name clear and simple
for idx, file in enumerate(csv_df_with_NN['filename']):
    csv_df_with_NN.at[idx,'filename'] = file.split('/')[-1].replace('.aag', '') # if and only if (idx==dataframe index) is true
    # if file contains "hwmcc20", then add "hwmcc20" to the row
    if "hwmcc20" in file:
        csv_df_with_NN.at[idx,'benchmark'] = "hwmcc20"
    else:
        csv_df_with_NN.at[idx,'benchmark'] = "hwmcc07"

for idx, file in enumerate(csv_df_without_NN['filename']):
    csv_df_without_NN.at[idx,'filename'] = file.split('/')[-1].replace('.aag', '')
    # if file contains "hwmcc20", then add "hwmcc20" to the row
    if "hwmcc20" in file:
        csv_df_with_NN.at[idx,'benchmark'] = "hwmcc20"
    else:
        csv_df_with_NN.at[idx,'benchmark'] = "hwmcc07"

# round the csv_df and replace the original csv_df
csv_df_with_NN = csv_df_with_NN.round(5)
csv_df_without_NN = csv_df_without_NN.round(5)

# remove duplicate in the csv_df_with_NN, sorting the row by the number of total frame and time reduced
csv_df_with_NN = csv_df_with_NN.sort_values(['Total Frame','Time reduce INF time'], ascending = True).drop_duplicates(subset = 'filename', keep = 'first').reset_index(drop=True)
csv_df_without_NN = csv_df_without_NN.sort_values(['Total Frame','Time Consuming'], ascending = False).drop_duplicates(subset = 'filename', keep = 'first').reset_index(drop=True)

# remove the last row in the csv_df_with_NN -> only for testing
# csv_df_with_NN.drop(csv_df_with_NN.tail(1).index,inplace=True)

# rename columns -> ready to merge
csv_df_without_NN.rename(columns = {'Total Frame':'Total Frame (without NN)', 'Number of clauses': 'clauses (without NN)', 'Time Consuming':'Time consuming (without NN)'}, inplace = True)
csv_df_with_NN.rename(columns = {'Time reduce INF time' : 'Time consuming (without INF time)','Number of clauses': 'clauses'},inplace = True)
# join the two dataframes (only if the filename is the same)
csv_df_without_NN = pd.merge(csv_df_without_NN,csv_df_with_NN, on = 'filename', how = 'left')
csv_df_without_NN = csv_df_without_NN.reindex(sorted(csv_df_without_NN.columns,reverse=True), axis=1)

# update the 'time reduce (percentage)' coloumn

for idx, _ in enumerate(csv_df_without_NN['Time consuming (without INF time)']):
    if (csv_df_without_NN.at[idx,'Time consuming (without INF time)']-csv_df_without_NN.at[idx,'Time consuming (without NN)']) / csv_df_without_NN.at[idx,'Time consuming (without NN)'] < 0:
        csv_df_without_NN.at[idx,'Time reduce'] = str(((csv_df_without_NN.at[idx,'Time consuming (without INF time)']-csv_df_without_NN.at[idx,'Time consuming (without NN)']) / csv_df_without_NN.at[idx,'Time consuming (without NN)'] * 100).round(1)) + '%'
    elif (csv_df_without_NN.at[idx,'Time consuming (without INF time)']-csv_df_without_NN.at[idx,'Time consuming (without NN)']) / csv_df_without_NN.at[idx,'Time consuming (without NN)'] == 0:
        csv_df_without_NN.at[idx,'Time reduce'] = '0%'
    elif (csv_df_without_NN.at[idx,'Time consuming (without INF time)']-csv_df_without_NN.at[idx,'Time consuming (without NN)']) / csv_df_without_NN.at[idx,'Time consuming (without NN)'] > 0:
        csv_df_without_NN.at[idx,'Time reduce'] = '+' + str(((csv_df_without_NN.at[idx,'Time consuming (without INF time)']-csv_df_without_NN.at[idx,'Time consuming (without NN)']) / csv_df_without_NN.at[idx,'Time consuming (without NN)'] * 100).round(1)) + '%'

# update all the value in the 'clauses', 'clauses (without NN)', 'Total Frame', ''Total Frame (without NN)'  column to int type without NaN
# csv_df_without_NN['clauses'] = csv_df_without_NN['clauses'].astype(int)

# update the 'clauses reduce' coloumn
for idx, _ in enumerate(csv_df_without_NN['clauses']):
    if csv_df_without_NN.at[idx,'clauses'] - csv_df_without_NN.at[idx,'clauses (without NN)'] < 0:
        csv_df_without_NN.at[idx,'Clauses changed'] =  (csv_df_without_NN.at[idx,'clauses'] - csv_df_without_NN.at[idx,'clauses (without NN)']).astype(int)
    elif csv_df_without_NN.at[idx,'clauses'] - csv_df_without_NN.at[idx,'clauses (without NN)'] > 0:
        # append '+' to the number
        csv_df_without_NN.at[idx,'Clauses changed'] = '+' + str((csv_df_without_NN.at[idx,'clauses'] - csv_df_without_NN.at[idx,'clauses (without NN)']).astype(int))
    else:
        csv_df_without_NN.at[idx,'Clauses changed'] = 0

# update the 'frames reduce' coloumn
for idx, _ in enumerate(csv_df_without_NN['Total Frame (without NN)']):
    if csv_df_without_NN.at[idx,'Total Frame'] - csv_df_without_NN.at[idx,'Total Frame (without NN)'] < 0:
        csv_df_without_NN.at[idx,'Frames changed'] =  (csv_df_without_NN.at[idx,'Total Frame'] - csv_df_without_NN.at[idx,'Total Frame (without NN)']).astype(int)
    elif csv_df_without_NN.at[idx,'Total Frame'] - csv_df_without_NN.at[idx,'Total Frame (without NN)'] > 0:
        csv_df_without_NN.at[idx,'Frames changed'] = '+' + str((csv_df_without_NN.at[idx,'Total Frame'] - csv_df_without_NN.at[idx,'Total Frame (without NN)']).astype(int))
    else:
        csv_df_without_NN.at[idx,'Frames changed'] = 0

# rename the dataframe as final_result
final_result = csv_df_without_NN

# remove the row that is NaN in Total Frame of final_result
final_result = final_result.dropna(subset = ['Total Frame'])

# Optional drop those rows that are NaN in passing ratio
final_result = final_result.dropna(subset = ['Passing Ratio'])

# drop rows that "Passing time" is zero
final_result = final_result[final_result['Passing Times'] != 0] 

# calculate the reduce ratio of cases that has been reduced by NN
reduce_success = sum(row['Total Frame (without NN)'] >= row['Total Frame'] or row['Time consuming (without NN)'] >= row['Time consuming (without INF time)'] for idx, row in final_result.iterrows())
reduce_frame_success = sum(row['Total Frame (without NN)'] > row['Total Frame'] for idx, row in final_result.iterrows())
high_prediction_success = sum(row['Prediction Thershold']>0.5 for _,row in final_result.iterrows())

print(f"{str((reduce_success/len(final_result))*100)}% of the cases have been reduced by NN")
print(f"{str((reduce_frame_success/len(final_result))*100)}% of the cases have converged earlier by applying NN")
print(f"{str((high_prediction_success/len(final_result))*100)} % of the cases have high success rate of NN prediction")

# Get sum of Total Frame (without NN) and Total Frame
print(f"Total Frame without NN: {final_result['Total Frame (without NN)'].sum()}")
print(f"Total Fram with NN: {final_result['Total Frame'].sum()}")
print(f"Total Frame reduce: {final_result['Total Frame (without NN)'].sum() - final_result['Total Frame'].sum()}")
print(f"Total Frame reduce percentage: {((final_result['Total Frame (without NN)'].sum() - final_result['Total Frame'].sum())/final_result['Total Frame (without NN)'].sum())*100}%")

# Get sum of Time consuming (without NN) and Time consuming (without INF time)
print(f"Time consuming (without NN): {final_result['Time consuming (without NN)'].sum()}")
print(f"Time consuming (without INF time): {final_result['Time consuming (without INF time)'].sum()}")
print(f"Time consuming reduce: {final_result['Time consuming (without NN)'].sum() - final_result['Time consuming (without INF time)'].sum()}")
print(f"Time consuming reduce percentage: {((final_result['Time consuming (without NN)'].sum() - final_result['Time consuming (without INF time)'].sum())/final_result['Time consuming (without NN)'].sum())*100}%")

# export to temporary csv file
with tempfile.TemporaryDirectory() as tmpdirname:
    print('created temporary directory', tmpdirname)
    final_result.to_csv(os.path.join(tmpdirname, 'result.csv'), index = False)
    print('exported to csv file')
    # copy the csv file to log directory with time stamp
    shutil.copy(os.path.join(tmpdirname, 'result.csv'), os.path.join("../log/", f'result_{time.strftime("%Y%m%d-%H%M%S")}.csv'))

# export the conclusion table
df_conclusion = pd.DataFrame(data=
            [['cases have been reduced by NN', ((reduce_success/len(final_result))*100)],
            ['cases have converged earlier by applying NN', ((reduce_frame_success/len(final_result))*100)],
            ['cases have high success rate of NN prediction', ((high_prediction_success/len(final_result))*100)],
            ['Total Frame reduce', (final_result['Total Frame (without NN)'].sum() - final_result['Total Frame'].sum())],
            ['Total Frame reduce percentage', (((final_result['Total Frame (without NN)'].sum() - final_result['Total Frame'].sum())/final_result['Total Frame (without NN)'].sum())*100).round(1)],
            ['Time consuming reduce', (final_result['Time consuming (without NN)'].sum() - final_result['Time consuming (without INF time)'].sum())],
            ['Time consuming reduce percentage', (((final_result['Time consuming (without NN)'].sum() - final_result['Time consuming (without INF time)'].sum())/final_result['Time consuming (without NN)'].sum())*100).round(1)]],

             columns = ['metric', 'value'])

# print df_conclusion as latex
df_conclusion = df_conclusion.round(1)
print(df_conclusion.to_latex(index=False))

'''
print final_result as latex
'''

# copy a dataframe to a new dataframe
final_result_latex = final_result.copy()

# drop the columns that are not needed
final_result_latex = final_result_latex.drop(columns = [ 
                                                        'Prediction Thershold',
                                                        'Time consuming (without NN)',
                                                         'Time Consuming', 
                                                         'Time consuming (without INF time)',
                                                         'Passing Times',
                                                         'Passing Ratio',
                                                         'Total Frame (without NN)',
                                                         'Total Frame',
                                                         'clauses',
                                                         'clauses (without NN)'])

# Drop columns that if 'Clauses changed', 'Frames changed' is both zero
final_result_latex = final_result_latex[(final_result_latex['Clauses changed'] != 0) | (final_result_latex['Frames changed'] != 0)]


# Change the type of coloumn of Total Frame to int
#final_result_latex['Total Frame'] = final_result_latex['Total Frame'].astype(int)
#final_result_latex['Total Frame (without NN)'] = final_result_latex['Total Frame (without NN)'].astype(int)
#final_result_latex['clauses'] = final_result_latex['clauses'].astype(int)

# fix the sequence of columns
final_result_latex = final_result_latex[['filename', 'benchmark', 'Clauses changed', 'Frames changed', 'Time reduce']]

# convert the string to int (in 'Clauses changed' and 'Frames changed')
final_result_latex['Clauses changed'] = final_result_latex['Clauses changed'].astype(int)
final_result_latex['Frames changed'] = final_result_latex['Frames changed'].astype(int)

# sort the dataframe by 'benchmark' and 'filename'
final_result_latex = final_result_latex.sort_values(by=['benchmark', 'filename'])

# add '+' to the postive value in 'Clauses changed' and 'Frames changed'
final_result_latex['Clauses changed'] = final_result_latex['Clauses changed'].apply(lambda x: f'+{x}' if x > 0 else x)
final_result_latex['Frames changed'] = final_result_latex['Frames changed'].apply(lambda x: f'+{x}' if x > 0 else x)

print(final_result_latex.to_latex(index=False))

display(final_result)
