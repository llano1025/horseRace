from data import data_scrapper as ds
import pandas as pd
import os

# save_past_race_result()
# df = ds.get_future_race_card()
# ds.get_horse_details()
# ds.save_past_race_result()

latest_file = ds.get_latest_csv('./data/horseInformation/')
dataframe_master = pd.read_csv(os.path.join('./data/horseInformation/', latest_file))
df = ds.reformat_horse_info(dataframe_master)
ds.save_horse_info(df)
print("hello world")


