import pandas as pd
from utils import label_encoding
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pandas_profiling import ProfileReport
import seaborn as sn
import matplotlib.pyplot as plt


# vechicles_csv = pd.read_csv("vehicles.csv")
# print(f"total number of rows in original csv: {len(vechicles_csv)}")

# input path where the new cleaned csv will be saved
vechicles_csv_processed_path = "vehicles-cleaned.csv"

features_to_consider = ["region", "price", "year", "manufacturer", "model", "condition", "cylinders", "fuel", "odometer",
                        "title_status", "transmission", "drive", "type", "paint_color", "state"]

# # drop rows that contain atleast one na value
# vechicles_csv = vechicles_csv.dropna(subset=features_to_consider)
# colms_to_remove = [col for col in vechicles_csv.columns if col not in features_to_consider]
# vechicles_csv.drop(colms_to_remove, axis=1, inplace=True)
# vechicles_csv.to_csv(vechicles_csv_processed_path)
# print(f"total number of rows in cleaned csv (no rows contain nan): {len(vechicles_csv)}")

vechicles_csv = pd.read_csv(vechicles_csv_processed_path)
print(f"total number of rows in cleaned csv (no rows contain nan): {len(vechicles_csv)}")

# filter year, consider after 1995
# vechicles_csv = vechicles_csv[vechicles_csv["year"].between(1995, 2021)]
vechicles_csv = vechicles_csv[vechicles_csv.year >= 2000]

# Removing outliers
# odometer_low = vechicles_csv.odometer.quantile(.01)
odometer_low = 4000
odometer_high = vechicles_csv.odometer.quantile(.95)
print(odometer_low, odometer_high)
vechicles_csv = vechicles_csv[vechicles_csv.odometer < odometer_high]
vechicles_csv = vechicles_csv[vechicles_csv.odometer > odometer_low]

# price_low = vechicles_csv.price.quantile(.045)
price_low = 1000
price_high = vechicles_csv.price.quantile(.988)
print(price_low, price_high)
vechicles_csv = vechicles_csv[vechicles_csv.price < price_high]
vechicles_csv = vechicles_csv[vechicles_csv.price > price_low]

print(f"total number of rows in cleaned csv (after removing percentile): {len(vechicles_csv)}")
object_cat_columns = ["region", "manufacturer", "model", "condition", "cylinders", "fuel", "title_status", "transmission",
                      "drive", "type", "paint_color", "state"]
# df, encoding_dict = label_encoding(vechicles_csv, object_cat_columns)
# print(encoding_dict)

for col in object_cat_columns:
    if col in vechicles_csv.columns:
        le = LabelEncoder()
        le.fit(list(vechicles_csv[col].astype(str).values))
        vechicles_csv[col] = le.transform(list(vechicles_csv[col].astype(str).values))

# remove unwanted columns and duplicates
vechicles_csv.reset_index(drop=True, inplace=True)
vechicles_csv.reset_index(drop=True)
vechicles_csv.drop(["Unnamed: 0"], axis=1, inplace=True)
vechicles_csv.drop_duplicates(inplace=True)
vechicles_csv.reset_index(drop=True, inplace=True)
vechicles_csv.reset_index(drop=True)
print(f"total number of rows in cleaned csv (after removing duplicates): {len(vechicles_csv)}")

price_df = vechicles_csv.pop("price")
vechicles_csv["price"] = price_df
# correlation
corrMatrix = vechicles_csv.corr()
fig = plt.figure(figsize=(15, 15))
sn.heatmap(corrMatrix, annot=True)
plt.title("Pearson Correlation")
# plt.show()
plt.savefig("./heatmap-correlation-filtered.png", dpi=256, bbox_inches='tight')

# detailed report of csv
profile = ProfileReport(vechicles_csv, title="Pandas Profiling Report", vars={'num':{'low_categorical_threshold': 0}})
profile.to_file("vehicles_csv_report-filtered.html")

print(vechicles_csv.columns)

# move price column to last
price_df = vechicles_csv.pop("price")
vechicles_csv["price"] = price_df


# all 14 features
df_numpy = vechicles_csv.to_numpy()
np.save(open("./datasets/filtered/features_14.npy", "wb"), df_numpy)

# top 12 features
features_to_drop = ["region", "paint_color"]
new_df = vechicles_csv.drop(features_to_drop, axis=1)
df_numpy = new_df.to_numpy()
np.save(open("./datasets/filtered/features_12.npy", "wb"), df_numpy)

# top 11 features
features_to_drop = ["region", "paint_color", "state"]
new_df = vechicles_csv.drop(features_to_drop, axis=1)
df_numpy = new_df.to_numpy()
np.save(open("./datasets/filtered/features_11.npy", "wb"), df_numpy)

# top 10 features
features_to_drop = ["region", "paint_color", "state", "type"]
new_df = vechicles_csv.drop(features_to_drop, axis=1)
df_numpy = new_df.to_numpy()
np.save(open("./datasets/filtered/features_10.npy", "wb"), df_numpy)

# top 8 features
features_to_drop = ["region", "paint_color", "state", "type", "manufacturer", "model"]
new_df = vechicles_csv.drop(features_to_drop, axis=1)
df_numpy = new_df.to_numpy()
np.save(open("./datasets/filtered/features_8.npy", "wb"), df_numpy)

# top 5 features for baseline comparison
baseline_features_to_consider = [ "price", "year", "manufacturer", "odometer", "transmission", "paint_color",]
colms_to_remove = [col for col in vechicles_csv.columns if col not in baseline_features_to_consider]
new_df = vechicles_csv.drop(colms_to_remove, axis=1)
df_numpy = new_df.to_numpy()
np.save(open("./datasets/filtered/features_5.npy", "wb"), df_numpy)
