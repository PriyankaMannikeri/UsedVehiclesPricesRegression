import pandas as pd
from utils import label_encoding
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pandas_profiling import ProfileReport
import seaborn as sn
import matplotlib.pyplot as plt
from category_encoders import *

# read raw csv file from Kaggle
# vehicles_csv = pd.read_csv("vehicles.csv")
# print(f"total number of rows in original csv: {len(vehicles_csv)}")

# input path where the new cleaned csv will be saved
vehicles_csv_processed_path = "vehicles-cleaned.csv"

# consider relevant and non-redundant features
features_to_consider = ["region", "price", "year", "manufacturer", "model", "condition", "cylinders", "fuel", "odometer",
                        "title_status", "transmission", "drive", "type", "paint_color", "state"]

# # drop rows that contain atleast one na value
# vehicles_csv = vehicles_csv.dropna(subset=features_to_consider)
# colms_to_remove = [col for col in vehicles_csv.columns if col not in features_to_consider]
# vehicles_csv.drop(colms_to_remove, axis=1, inplace=True)
# vehicles_csv.to_csv(vehicles_csv_processed_path)
# print(f"total number of rows in cleaned csv (no rows contain nan): {len(vehicles_csv)}")

vehicles_csv = pd.read_csv(vehicles_csv_processed_path)
print(f"total number of rows in cleaned csv (no rows contain nan): {len(vehicles_csv)}")

# Removing outliers

# filter year, consider after 2000
vehicles_csv = vehicles_csv[vehicles_csv.year >= 2000]

# clip odometer values
# odometer_low = vehicles_csv.odometer.quantile(.01) #1%
odometer_low = 4000
odometer_high = vehicles_csv.odometer.quantile(.95) #95%
print(odometer_low, odometer_high)
vehicles_csv = vehicles_csv[vehicles_csv.odometer < odometer_high]
vehicles_csv = vehicles_csv[vehicles_csv.odometer > odometer_low]

# clip price values
# price_low = vehicles_csv.price.quantile(.045)
price_low = 1000
price_high = vehicles_csv.price.quantile(.988)
print(price_low, price_high)
vehicles_csv = vehicles_csv[vehicles_csv.price < price_high]
vehicles_csv = vehicles_csv[vehicles_csv.price > price_low]

print(f"total number of rows in cleaned csv (after removing percentile): {len(vehicles_csv)}")

# 12 categorical columns
object_cat_columns = ["region", "manufacturer", "model", "condition", "cylinders", "fuel", "title_status", "transmission",
                      "drive", "type", "paint_color", "state"]

# df, encoding_dict = label_encoding(vehicles_csv, object_cat_columns)
# print(encoding_dict)

# move price column to last (for heatmap vis)
price_df = vehicles_csv.pop("price")
vehicles_csv["price"] = price_df

# ordered label encoding
# mapping = [
#     {
#         "col": "condition",
#         "mapping":
#             {
#                 "new": 0,
#                 "like new": 1,
#                 "excellent": 2,
#                 "good": 3,
#                 "fair": 4,
#                 "salvage": 5,
#             }
#     }
# ]
# enc = OrdinalEncoder(cols=["condition"], mapping=mapping).fit(vehicles_csv)
# vehicles_csv = enc.transform(vehicles_csv)

# alphabetical label encoding
for col in object_cat_columns:
    if col in vehicles_csv.columns:
        le = LabelEncoder()
        le.fit(list(vehicles_csv[col].astype(str).values))
        vehicles_csv[col] = le.transform(list(vehicles_csv[col].astype(str).values))

# one-hot encoding
# vehicles_csv = pd.get_dummies(vehicles_csv, columns=["condition", "cylinders", "fuel", "title_status", "transmission",
#                                                        "drive", "type"])
# for col in ["manufacturer", "model"]:
#     if col in vehicles_csv.columns:
#         le = LabelEncoder()
#         le.fit(list(vehicles_csv[col].astype(str).values))
#         vehicles_csv[col] = le.transform(list(vehicles_csv[col].astype(str).values))


# remove unwanted columns and duplicates
vehicles_csv.reset_index(drop=True, inplace=True)
vehicles_csv.reset_index(drop=True)
vehicles_csv.drop(["Unnamed: 0"], axis=1, inplace=True)
vehicles_csv.drop_duplicates(inplace=True)
vehicles_csv.reset_index(drop=True, inplace=True)
vehicles_csv.reset_index(drop=True)
print(f"total number of rows in cleaned csv (after removing duplicates): {len(vehicles_csv)}")

# correlation after featurization
corrMatrix = vehicles_csv.corr()
fig = plt.figure(figsize=(15, 15))
sn.heatmap(corrMatrix, annot=True)
plt.title("Pearson Correlation")
plt.show()
# plt.savefig("./heatmap-correlation-filtered-onehot.png", dpi=256, bbox_inches='tight')

# Pandas profiling: detailed report of csv in html format
profile = ProfileReport(vehicles_csv, title="Pandas Profiling Report", vars={'num':{'low_categorical_threshold': 0}})
profile.to_file("vehicles_csv_report-filtered.html")

print(vehicles_csv.columns)

# move price column to last for numpy
price_df = vehicles_csv.pop("price")
vehicles_csv["price"] = price_df

# all 14 features
# df_numpy = vehicles_csv.to_numpy()
# np.save(open("./datasets/filtered/features_14.npy", "wb"), df_numpy)
#
# # top 12 features
# features_to_drop = ["region", "paint_color"]
# new_df = vehicles_csv.drop(features_to_drop, axis=1)
# df_numpy = new_df.to_numpy()
# np.save(open("./datasets/filtered/features_12.npy", "wb"), df_numpy)

# top 11 features
features_to_drop = ["region", "paint_color", "state"]
new_df = vehicles_csv.drop(features_to_drop, axis=1)
df_numpy = new_df.to_numpy()
# np.save(open("./datasets/filtered/features_11.npy", "wb"), df_numpy)
# np.save(open("./datasets/onehot-encoded/features_48.npy", "wb"), df_numpy)
np.save(open("./datasets/ordered-label-encoding/features_11.npy", "wb"), df_numpy)

# top 10 features
# features_to_drop = ["region", "paint_color", "state", "type"]
# new_df = vehicles_csv.drop(features_to_drop, axis=1)
# df_numpy = new_df.to_numpy()
# np.save(open("./datasets/filtered/features_10.npy", "wb"), df_numpy)

# top 8 features
# features_to_drop = ["region", "paint_color", "state", "type", "manufacturer", "model"]
# new_df = vehicles_csv.drop(features_to_drop, axis=1)
# df_numpy = new_df.to_numpy()
# np.save(open("./datasets/filtered/features_8.npy", "wb"), df_numpy)
#
# # top 5 features for baseline comparison
# baseline_features_to_consider = [ "price", "year", "manufacturer", "odometer", "transmission", "paint_color",]
# colms_to_remove = [col for col in vehicles_csv.columns if col not in baseline_features_to_consider]
# new_df = vehicles_csv.drop(colms_to_remove, axis=1)
# df_numpy = new_df.to_numpy()
# np.save(open("./datasets/filtered/features_5.npy", "wb"), df_numpy)
