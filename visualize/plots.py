import os

import pandas as pd

project_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(project_path, "data")

df = pd.read_csv(os.path.join(data_path, "age_dist.csv"))
pass