import pandas as pd
from neo4j import GraphDatabase

# Load the CSV file
file_path = 'D:/Git/horseRace/data/pastRaceResult//2024.06.17_race_result.csv'
df = pd.read_csv(file_path)
uri = "bolt://localhost:7687"  # Update this with your Neo4j instance URI
user = "neo4j"  # Your Neo4j username
password = "password"  # Your Neo4j password

driver = GraphDatabase.driver(uri, auth=(user, password))


def create_race(tx, race_name, date, location, course, distance, condition, race_class):
    tx.run("CREATE (r:Race {race_name: $race_name, date: $date, location: $location, course: $course, distance: "
           "$distance, condition:"
           "$condition, _class: $_class})",
           race_name=race_name, date=date, location=location, course=course, distance=distance, condition=condition, _class=race_class)


def create_horse(tx, horse_name):
    tx.run("CREATE (h:Horse {horse_name: $horse_name})", horse_name=horse_name)


def create_jockey(tx, jockey_name):
    tx.run("CREATE (j:Jockey {jockey_id: $jockey_id, jockey_name: $jockey_name})", jockey_name=jockey_name)


def create_trainer(tx, trainer_name):
    tx.run("CREATE (t:Trainer {trainer_id: $trainer_id, trainer_name: $trainer_name})", trainer_name=trainer_name)


def create_race_result_relationship(tx, race_name, horse_name, jockey_name, trainer_name, placement, draw, horse_weight):
    tx.run("MATCH (r:Race {race_name: $race_name}), (h:Horse {horse_name: $horse_name}), (j:Jockey {jockey_name: "
           "$jockey_name}), (t:Trainer {trainer_name: $trainer_name})"
           "CREATE (h)-[:PARTICIPATED_IN {placement: $placement, draw: $draw, horse_weight: $ horse_weight}]->(r), "
           "(j)-[:RIDDEN]->(h),"
           "(t)-[:TRAINED]->(h)",
           race_id=race_name, horse_name=horse_name, jockey_id=jockey_name, trainer_id=trainer_name, placement=placement,
           draw=draw, horse_weight=horse_weight)


def load_data_to_neo4j(df):
    with driver.session() as session:
        for index, row in df.iterrows():
            date = row['Race Date']
            location = row['Race Location']
            course = row['Course']
            distance = row['Distance']
            condition = row['Going']
            race_class = row['Class']
            race_name = f"{date} - {location} - {course} - {distance} - {race_class}"

            horse_id = row['Brand No.']
            horse_name = row['Horse']
            jockey_name = row['Jockey']
            trainer_name = row['Trainer']

            placement = row['Pla.']
            draw = row['Dr.']
            horse_weight = row['Declar. Horse Wt.']

            session.write_transaction(create_race, race_name, date, location, course, distance, condition, race_class)
            session.write_transaction(create_horse, horse_name)
            session.write_transaction(create_jockey, jockey_name)
            session.write_transaction(create_trainer, trainer_name)
            session.write_transaction(create_race_result_relationship, race_name, horse_name, jockey_name, trainer_name,
                                      placement, draw, horse_weight)


# Load the data into Neo4j
load_data_to_neo4j(df)
