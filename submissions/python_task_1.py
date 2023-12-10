import pandas as pd
import numpy as np 
import os

file_path = './datasets/dataset-1.csv'
df = pd.read_csv('./datasets/dataset-1.csv')
def generate_car_matrix(df)->pd.DataFrame:
    generate_car_matrix = df.pivot(index='id_1', columns='id_2', values='car')
    diagonal_indices = generate_car_matrix.index.intersection(generate_car_matrix.columns)

    # Replace diagonal values with 0 using NumPy
    generate_car_matrix.values[np.diag_indices(len(diagonal_indices))] = 0
    return generate_car_matrix 
result = generate_car_matrix(df)
print(result)





def df2(df):
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=['low', 'medium', 'high'])

    type_count = df['car_type'].value_counts().to_dict()
    
    type_count = dict(sorted(type_count.items()))

    return type_count
dataset = pd.read_csv('./datasets/dataset-1.csv')
result = df2(dataset)
print(result)



def get_bus_indexes(df)->list:
    mean_bus = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes
result = get_bus_indexes(df)
print(result)


def filter_routes(df)->list:
    # Group by 'route' and calculate the average of 'truck' column
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' column is greater than 7
    filtered_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Filter the DataFrame based on the filtered routes
    filtered_df = df[df['route'].isin(filtered_routes)]
    
    # Sort the DataFrame by 'route' column
    filtered_df = filtered_df.sort_values(by='route')

    return filtered_df[['route', 'truck']]  # Return specific columns
result = filter_routes(df)
print(result)

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
modified_df = multiply_matrix(generate_car_matrix(df))
print(modified_df)

