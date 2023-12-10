#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv('dataset-1.csv')


# In[3]:


df.columns


# In[4]:


df.shape


# # 1

# In[5]:


df1=df.pivot(index='id_1', columns='id_2', values='car')


# In[6]:


# Get the diagonal indices
diagonal_indices = df1.index.intersection(df1.columns)

# Replace diagonal values with 0 using NumPy
df1.values[np.diag_indices(len(diagonal_indices))] = 0
df1


# In[39]:


#


# In[7]:


def df2(df):
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=['low', 'medium', 'high'])

    type_count = df['car_type'].value_counts().to_dict()
    
    type_count = dict(sorted(type_count.items()))

    return type_count


# In[8]:


dataset = pd.read_csv('dataset-1.csv')


# In[9]:


result = df2(dataset)
print(result)


# In[48]:


#3


# In[10]:


def get_bus_indexes(df):
    mean_bus = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes


# In[11]:


result = get_bus_indexes(df)
print(result)


# In[51]:


#4


# In[12]:


def filter_routes(df):
    # Group by 'route' and calculate the average of 'truck' column
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' column is greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Filter the DataFrame based on the filtered routes
    filtered_df = df[df['route'].isin(filtered_routes)]
    
    # Sort the DataFrame by 'route' column
    filtered_df = filtered_df.sort_values(by='route')

    return filtered_df[['route', 'truck']]  # Return specific columns


# In[72]:


result = filter_routes(df)
print(result)


# In[73]:


#5


# In[13]:


def multiply_matrix(df):
    # Iterate through each value in the DataFrame
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iloc[i, j]
            if value > 20:
                df.iloc[i, j] = round(value * 0.75, 1)
            else:
                df.iloc[i, j] = round(value * 1.25, 1)
    
    return df


# In[14]:


modified_df = multiply_matrix(df1.copy())
print(modified_df)


# In[34]:


#2_1


# In[ ]:


# all pair distance


# In[181]:


df = pd.read_csv('dataset-3.csv')
distance_matrix=df.pivot(index='id_start', columns='id_end', values='distance')
l1=[0]
l1.extend([9.7])
l1.extend([np.nan]*(distance_matrix.shape[0]-2))

distance_matrix.insert(0, 1001400, l1)
distance_matrix=distance_matrix.fillna(0)
distance_matrix_all=distance_matrix.copy()


# In[182]:


#using diagonal value fill the above value,all pair distance matrix with cummulative distance


# In[183]:


#cl=[i for i in distance_matrix.columns.values]
for i in distance_matrix_all.index:
      for j in distance_matrix_all.columns:
        if j>=i:
            column_index = distance_matrix_all.columns.get_loc(j)
            k = distance_matrix_all.columns[column_index-1]
            #print(i,k,j)
            if k in distance_matrix_all.index:
                v=distance_matrix_all.loc[i,k]+distance_matrix_all.loc[k,j]
                distance_matrix_all.loc[i,j]=v
                distance_matrix_all.loc[j,i]=v
distance_matrix_all


# In[ ]:


#unpivot above result


# In[185]:


unpivoted_df = distance_matrix_all.reset_index().melt(id_vars='id_start', var_name='id_end', value_name='distance')

print(unpivoted_df)


# In[186]:


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


# In[191]:


reference_id = 1001400
result = find_ids_within_ten_percentage_threshold(unpivoted_df, reference_id)
print(result)


# In[ ]:


#4


# In[197]:


def calculate_toll_rate(unpivoted_df):
    # Create columns with rate coefficients for different vehicle types
    unpivoted_df['moto'] = unpivoted_df['distance'] * 0.8
    unpivoted_df['car'] = unpivoted_df['distance'] * 1.2
    unpivoted_df['rv'] = unpivoted_df['distance'] * 1.5
    unpivoted_df['bus'] = unpivoted_df['distance'] * 2.2
    unpivoted_df['truck'] = unpivoted_df['distance'] * 3.6

    return unpivoted_df



# In[198]:


result_with_toll_rates = calculate_toll_rate(unpivoted_df)
print(result_with_toll_rates)



# In[199]:


from datetime import datetime, time, timedelta


# In[234]:


import pandas as pd
import datetime

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


# In[235]:


result = calculate_time_based_toll_rates(unpivoted_df)
print(result)



# In[ ]:




