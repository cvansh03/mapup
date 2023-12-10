import pandas as pd
import numpy as np 
import os
from datetime import datetime, time, timedelta

file_path = './datasets/dataset-3.csv'
df = pd.read_csv('./datasets/dataset-3.csv')


def calculate_distance_matrix(df)->pd.DataFrame():
    calculate_distance_matrix=df.pivot(index='id_start', columns='id_end', values='distance')
    l1=[0]
    l1.extend([9.7])
    l1.extend([np.nan]*(calculate_distance_matrix.shape[0]-2))
    calculate_distance_matrix.insert(0, 1001400, l1)
    calculate_distance_matrix=calculate_distance_matrix.fillna(0)
    for i in calculate_distance_matrix.index:
      for j in calculate_distance_matrix.columns:
        if j>=i:
            column_index = calculate_distance_matrix.columns.get_loc(j)
            k = calculate_distance_matrix.columns[column_index-1]
            #print(i,k,j)
            if k in calculate_distance_matrix.index:
                v=calculate_distance_matrix.loc[i,k]+calculate_distance_matrix.loc[k,j]
                calculate_distance_matrix.loc[i,j]=v
                calculate_distance_matrix.loc[j,i]=v
print(calculate_distance_matrix)


def unpivoted_df(df)->pd.DataFrame():
    unpivoted_df = calculate_distance_matrix.reset_index().melt(id_vars='id_start', var_name='id_end', value_name='distance')

print(unpivoted_df)


def find_ids_within_ten_percentage_threshold(dataframe, reference_value):
    # Filter the DataFrame for the reference_value
    reference_data = dataframe[dataframe['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    avg_distance = reference_data['distance'].mean()

    # Calculate the threshold range (10% of the average distance)
    threshold = 0.1 * avg_distance

    # Filter values within the threshold without using loops
    within_threshold = dataframe[
        (dataframe['distance'] >= avg_distance - threshold) &
        (dataframe['distance'] <= avg_distance + threshold)
    ]

    # Get unique id_start values within the threshold
    filtered_ids = sorted(within_threshold['id_start'].unique())

    return filtered_ids
reference_id = 1001400
result = find_ids_within_ten_percentage_threshold(unpivoted_df, reference_id)
print(result)


def calculate_toll_rate(unpivoted_df):
    # Create columns with rate coefficients for different vehicle types
    unpivoted_df['moto'] = unpivoted_df['distance'] * 0.8
    unpivoted_df['car'] = unpivoted_df['distance'] * 1.2
    unpivoted_df['rv'] = unpivoted_df['distance'] * 1.5
    unpivoted_df['bus'] = unpivoted_df['distance'] * 2.2
    unpivoted_df['truck'] = unpivoted_df['distance'] * 3.6

    return unpivoted_df
result_with_toll_rates = calculate_toll_rate(unpivoted_df)
print(result_with_toll_rates)


def calculate_time_based_toll_rates(dataframe):
    # Define a function to calculate the discount factor based on time
    def calculate_discount_factor(time):
        if time < datetime.time(10, 0, 0):
            return 0.8
        elif datetime.time(10, 0, 0) <= time < datetime.time(18, 0, 0):
            return 1.2
        else:
            return 0.8 if time <= datetime.time(23, 59, 59) else 0.7

    # Create start and end times for each time interval
    start_times = [datetime.time(0, 0, 0), datetime.time(10, 0, 0), datetime.time(18, 0, 0)]
    end_times = [datetime.time(9, 59, 59), datetime.time(17, 59, 59), datetime.time(23, 59, 59)]

    # Create a list of days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    result_list = []

    # Iterate through each unique id_start and id_end pair
    for _, group in dataframe.groupby(['id_start', 'id_end']):
        id_start, id_end = _
        for day in days_of_week:
            for start_time, end_time in zip(start_times, end_times):
                # Create a new row with time-based toll rates
                new_row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'end_day': day,
                    'start_time': start_time,
                    'end_time': end_time,
                    'vehicle': calculate_discount_factor(start_time)
                }
                result_list.append(new_row)

    # Convert the list of dictionaries to a DataFrame
    result = pd.DataFrame(result_list)

    # Convert start_day and end_day to proper case
    result['start_day'] = result['start_day'].str.capitalize()
    result['end_day'] = result['end_day'].str.capitalize()

    return result

result = calculate_time_based_toll_rates(unpivoted_df)
print(result)
