import pandas as pd
from utils import label_encoding
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas_profiling import ProfileReport
import seaborn as sn
import matplotlib.pyplot as plt


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
# print(f"total number of rows in cleaned csv (no rows contain nan): {len(vechicles_csv)}")

vechicles_csv = pd.read_csv(vechicles_csv_processed_path)
print(f"total number of rows in cleaned csv (no rows contain nan): {len(vechicles_csv)}")

# Removing outliers
odometer_low = vechicles_csv.odometer.quantile(.01)
odometer_high = vechicles_csv.odometer.quantile(.95)
print(odometer_low, odometer_high)
# vechicles_csv.odometer = vechicles_csv.odometer.clip(lower=odometer_low, upper=odometer_high)
vechicles_csv = vechicles_csv[vechicles_csv.odometer < odometer_high]
vechicles_csv = vechicles_csv[vechicles_csv.odometer > odometer_low]

price_low = vechicles_csv.price.quantile(.045)
price_high = vechicles_csv.price.quantile(.988)
print(price_low, price_high)
# vechicles_csv.price = vechicles_csv.price.clip(lower=price_low, upper=price_high)
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

# correlation
corrMatrix = vechicles_csv.corr()
fig = plt.figure(figsize=(15, 15))
sn.heatmap(corrMatrix, annot=True)
plt.title("Pearson Correlation")
# plt.show()
plt.savefig("./heatmap-correlation.png", dpi=256, bbox_inches='tight')

profile = ProfileReport(vechicles_csv, title="Pandas Profiling Report", vars={'num':{'low_categorical_threshold': 0}})
profile.to_file("vehicles_csv_report.html")

"""
# and scaler function:
# use after correlation
features_to_scale = ["region", "year", "manufacturer", "model", "condition", "cylinders", "fuel", "odometer",
                        "title_status", "transmission", "drive", "type", "paint_color", "state"]
scaler = StandardScaler()
vechicles_csv = pd.DataFrame(scaler.fit_transform(vechicles_csv), columns=features_to_scale)

# for each feature print min, max, mean std, unique

# # TODO check the min. max values of numpy array (some car selling prices are 0!!)
df_numpy = df.to_numpy()
# print(df_numpy.shape)
df_numpy = np.delete(df_numpy, 0, 1)
df_numpy[:, [1, 14]] = df_numpy[:, [14, 1]]
# print(df_numpy.shape)
#
np.save(open("features.npy", "wb"), df_numpy)
"""
