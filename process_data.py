import pandas as pd
from utils import label_encoding
import numpy as np


vechicles_csv = pd.read_csv("vehicles.csv")
print(f"total number of rows in original csv: {len(vechicles_csv)}")

# input path where the new cleaned csv will be saved
vechicles_csv_processed_path = "vehicles-cleaned.csv"

features_to_consider = ["region", "price", "year", "manufacturer", "model", "condition", "cylinders", "fuel", "odometer",
                        "title_status", "transmission", "drive", "type", "paint_color", "state"]

# # drop rows that contain atleast one na value
vechicles_csv = vechicles_csv.dropna(subset=features_to_consider)
colms_to_remove = [col for col in vechicles_csv.columns if col not in features_to_consider]
vechicles_csv.drop(colms_to_remove, axis=1, inplace=True)
vechicles_csv.to_csv(vechicles_csv_processed_path)
print(f"total number of rows in cleaned csv (no rows contain nan): {len(vechicles_csv)}")

vechicles_csv = pd.read_csv(vechicles_csv_processed_path)
print(f"total number of rows in cleaned csv (no rows contain nan): {len(vechicles_csv)}")


object_cat_columns = ["region", "manufacturer", "model", "condition", "cylinders", "fuel", "title_status", "transmission",
                      "drive", "type", "paint_color", "state"]
df, encoding_dict = label_encoding(vechicles_csv, object_cat_columns)
print(encoding_dict)

# # TODO check the min. max values of numpy array (some car selling prices are 0!!)
df_numpy = df.to_numpy()
# print(df_numpy.shape)
df_numpy = np.delete(df_numpy, 0, 1)
df_numpy[:, [1, 14]] = df_numpy[:, [14, 1]]
print(df_numpy.shape)
#
np.save(open("features.npy", "wb"), df_numpy)
