"""
This module implements KNN algorithm
"""
import pandas as pd
from pandas import DataFrame
from collections import Counter
from typing import List, Dict, Tuple
from typing import Any


def read_dataset(file_path: str) -> pd.DataFrame:
    dataset = pd.read_csv(file_path)
    return dataset

Coordinate = Tuple[int, int]
def euclidean_distance(x: Coordinate, y: Coordinate) -> float :
    """
        Returns euclidean distance between two points

        euclidean distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    """

    distance = ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5
    print(f"distance = (({x[0]} - {y[0]}) ** 2) + (({x[1]} - {y[1]}) ** 2) ** 0.5 = {distance}")
    return distance


def filter_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
        returns the subsets of given dataset which ...
    """
    filtered_dataset = dataset.iloc[:, 1:3]
    return filtered_dataset


def dataframe_to_tuple(df: DataFrame) -> Tuple[Any, Any]:
    new_tuple = tuple(df.itertuples(index=False, name=None))
    return new_tuple


def calculate_distance(data_tuple, x1: int, x2: int, dataset) -> List[float]:
    new_data = (x1, x2)
    calculated_distances = []
    for value_pair in data_tuple:
        dist = euclidean_distance(new_data, value_pair)
        calculated_distances.append(dist)
    return calculated_distances


def updated_disances_df(calculated_distance,dataset):
    dataset_with_distances = dataset.copy()
    dataset_with_distances['distances'] = calculated_distance
    return dataset_with_distances
    

def select_top_k_distances(dataset_with_distances: DataFrame, k=3):
    """
        Returns nearest three neighbours
    """
    top_k_distances = dataset_with_distances['distances'].nlargest(k)

    print("\nTop K Distances\n", top_k_distances, "\n")
   
    return top_k_distances

    

def calculate_class_wise_index_count(top_k_distances, dataset_with_distances):
    classes = []
    for index, value in top_k_distances.iteritems():
        print("index", index)
        row = dataset_with_distances.iloc[index]
        row_class = row["Class of Sport"]
        classes.append(row_class)
    
    class_wise_index_count = Counter(classes)
    print("\nclasses count: ", class_wise_index_count)
    return class_wise_index_count



def check_most_similar_class(class_wise_index_count):
    max_class_count = max(class_wise_index_count, key=class_wise_index_count. get)
    return max_class_count


def k_nearest_neighbors():
    """
        This is the main entrypoint, orchestrating the KNN algorithm
    """

    dataset = read_dataset('dataset.csv')
    print("\nDataset\n",dataset, "\n")

    new_point_x = 45
    new_point_y = 0
    dataset_len = len(dataset)
    k = int(input("Enter value of K: "))
    
    while k >= dataset_len:
        print(f"Length of Dataset is {dataset_len}. \
             Please set the value of K less then the\
             length of rows in dataset")
        k = int(input("\nEnter value of K: "))
    
    print("\nNew test points: ",new_point_x," & ",new_point_y,"\n" )

    filtered_columns = filter_dataset(dataset)
    print("\nfiltered age and gender columns from given dataset: \n\n", filtered_columns,"\n")
    data_tuple = dataframe_to_tuple(filtered_columns)
    print("\nData Tuples of age and gender from given dataset\n\n", data_tuple, "\n")
    
    print("\nCalculated Distances\n")
    calculated_distance = calculate_distance(data_tuple, new_point_x, new_point_y, dataset)
    print("\n")
    dataset_with_distances = updated_disances_df(calculated_distance, dataset)
    print("\ndataset with calculated distances from new data points\n\n", dataset_with_distances, "\n")
    
    top_k_distances = select_top_k_distances(dataset_with_distances, k)
    
    class_wise_index_count = calculate_class_wise_index_count(top_k_distances, dataset_with_distances)
    
    
    most_similar_class = check_most_similar_class(class_wise_index_count)
    print(f"\nThis new data belongs to class: ", most_similar_class)

    
if __name__ == '__main__':
    k_nearest_neighbors()
    print(__doc__)