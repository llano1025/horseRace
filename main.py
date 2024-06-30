from data import data_scrapper as ds
import pandas as pd
import os

# save_past_race_result()
# df = ds.get_future_race_card()
ds.get_horse_details()
# ds.save_past_race_result()

horse_df = ds.get_horse_details()
