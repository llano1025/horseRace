from data import data_scrapper as ds
from model import neo4jw as nw


#  Update the race result
ds.save_past_race_result()

#  Update horse database
ds.get_horse_details()

#  Update neo4j database
nw.load_to_neo4j()

#  Get future race card
df = ds.get_future_race_card()
