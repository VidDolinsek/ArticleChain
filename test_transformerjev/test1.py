import numpy as np
from test_transformerjev.read_yaml_data import read_yaml_exclude_urls 


file = read_yaml_exclude_urls("testing_dataset.yaml")

print(len(file['title']))
