import pandas as pd
from IPython.display import display
import os.path

# Get .csv file list in log folder
csv_file_lst_with_NN = [f"../log/small_subset_experiment_with_NN_subset_{str(i)}.csv" for i in range(11) if os.path.isfile(f"../log/small_subset_experiment_with_NN_subset_{str(i)}.csv")]
csv_file_lst_without_NN = [f"../log/small_subset_without_NN_subset_{str(i)}.csv" for i in range(11) if os.path.isfile(f"../log/small_subset_without_NN_subset_{str(i)}.csv")]

# walk through the csv_file_with_NN and csv_file_without_NN in ../log/
csv_data_with_NN_frame_lst = [pd.read_csv(csv_with_NN, low_memory = False, index_col=False) for csv_with_NN in csv_file_lst_with_NN[:]]
csv_data_without_NN_frame_lst = [pd.read_csv(csv_without_NN, low_memory = False, index_col=False) for csv_without_NN in csv_file_lst_without_NN[:]]

# concat all the dataframe with ignore index (which is important!)
csv_df_with_NN = pd.concat(csv_data_with_NN_frame_lst,ignore_index=True)
csv_df_without_NN = pd.concat(csv_data_without_NN_frame_lst,ignore_index=True)

# update the value in the first row, make file name clear and simple
for idx, file in enumerate(csv_df_with_NN['filename']):
    csv_df_with_NN.at[idx,'filename'] = file.split('/')[-1].replace('.aag', '') # if and only if (idx==dataframe index) is true
for idx, file in enumerate(csv_df_without_NN['filename']):
    csv_df_without_NN.at[idx,'filename'] = file.split('/')[-1].replace('.aag', '')

# round the csv_df and replace the original csv_df
csv_df_with_NN = csv_df_with_NN.round(5)
csv_df_without_NN = csv_df_without_NN.round(5)

# remove duplicate in the csv_df_with_NN, sorting the row by the number of total frame and time reduced
csv_df_with_NN = csv_df_with_NN.sort_values(['Total Frame','Time reduce INF time'], ascending = True).drop_duplicates(subset = 'filename', keep = 'first').reset_index(drop=True)

# remove the last row in the csv_df_with_NN -> only for testing
# csv_df_with_NN.drop(csv_df_with_NN.tail(1).index,inplace=True)

# rename columns -> ready to merge
csv_df_without_NN.rename(columns = {'Total Frame':'Total Frame (without NN)', 'Number of clauses': 'Number of clauses (without NN)', 'Time Consuming':'Time consuming (without NN)'}, inplace = True)
csv_df_with_NN.rename(columns = {'Time reduce INF time' : 'Time consuming (without INF time)'},inplace = True)
# join the two dataframes (only if the filename is the same)
csv_df_without_NN = pd.merge(csv_df_without_NN,csv_df_with_NN, on = 'filename', how = 'left')
csv_df_without_NN = csv_df_without_NN.reindex(sorted(csv_df_without_NN.columns,reverse=True), axis=1)


# remove the row that is NaN in Total Frame of csv_df_without_NN
csv_df_without_NN = csv_df_without_NN.dropna(subset = ['Total Frame'])

# calculate the reduce ratio of cases that has been reduced by NN
reduce_success = sum(row['Total Frame (without NN)'] >= row['Total Frame'] or row['Time consuming (without NN)'] >= row['Time consuming (without INF time)'] for idx, row in csv_df_without_NN.iterrows())
reduce_frame_success = sum(row['Total Frame (without NN)'] > row['Total Frame'] for idx, row in csv_df_without_NN.iterrows())
high_prediction_success = sum(row['Prediction Thershold']>0.5 for _,row in csv_df_without_NN.iterrows())

print(f"{str(reduce_success/len(csv_df_without_NN))}% of the cases have been reduced by NN")
print(f"{str(reduce_frame_success/len(csv_df_without_NN))}% of the cases have converged earlier by applying NN")
print(f"{str(high_prediction_success/len(csv_df_without_NN))} % of the cases have high success rate of NN prediction")

#display(csv_df_without_NN)