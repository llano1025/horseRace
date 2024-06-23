from data import data_scrapper as ds
import pandas as pd
import os

# save_past_race_result()
# df = ds.get_future_race_card()
# ds.save_past_race_result()

horse_name = "KARATE EXPRESS"  # Replace with the name of the horse you are searching for
horse_url = ds.search_horse_by_name(horse_name)

if horse_url:
    horse_name, horse_details = ds.get_horse_details(horse_url)
    print(f"Horse Name: {horse_name}")
    for key, value in horse_details.items():
        print(f"{key}: {value}")
else:
    print("Horse not found.")