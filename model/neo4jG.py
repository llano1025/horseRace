import pandas as pd
from neo4j import GraphDatabase

# Load the CSV file
file_path = 'G:/Git/horseRace/data/pastRaceResult/2024.06.17_race_result.csv'
df = pd.read_csv(file_path)
uri = "bolt://localhost:7687"  # Update this with your Neo4j instance URI
user = "neo4j"  # Your Neo4j username
password = "password"  # Your Neo4j password

driver = GraphDatabase.driver(uri, auth=(user, password))


# Functions to create nodes and relationships
def create_race(tx, races):
    for race in races:
        tx.run("""
        MERGE (r:Race {race_name: $race_name})
        ON CREATE SET r.date = $date, r.location = $location, 
                      r.course = $course, r.distance = $distance, 
                      r.condition = $condition, r.class = $_class
        """, race_name=race['race_name'], date=race['date'], location=race['location'],
               course=race['course'], distance=race['distance'], condition=race['condition'], _class=race['class'])


def create_horse(tx, horses):
    for horse in horses:
        tx.run("""
        MERGE (h:Horse {horse_name: $horse_name})
        """, horse_name=horse['horse_name'])


def create_jockey(tx, jockeys):
    for jockey in jockeys:
        tx.run("""
        MERGE (j:Jockey {jockey_name: $jockey_name})
        """, jockey_name=jockey['jockey_name'])


def create_trainer(tx, trainers):
    for trainer in trainers:
        tx.run("""
        MERGE (t:Trainer {trainer_name: $trainer_name})
        """, trainer_name=trainer['trainer_name'])


def create_race_result_relationship(tx, relationships):
    for relationship in relationships:
        tx.run("""
        MATCH (r:Race {race_name: $race_name}), (h:Horse {horse_name: $horse_name}), 
              (j:Jockey {jockey_name: $jockey_name}), (t:Trainer {trainer_name: $trainer_name})
        MERGE (h)-[:PARTICIPATED_IN {placement: $placement, draw: $draw, horse_weight: $horse_weight}]->(r)
        MERGE (j)-[:RIDDEN]->(h)
        MERGE (t)-[:TRAINED]->(h)
        """, race_name=relationship['race_name'], horse_name=relationship['horse_name'],
               jockey_name=relationship['jockey_name'], trainer_name=relationship['trainer_name'],
               placement=relationship['placement'], draw=relationship['draw'],
               horse_weight=relationship['horse_weight'])


# Helper function to split data into batches
def split_into_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# Main function to execute the batching process
def load_data_to_neo4j(df, batch_size=100):
    with driver.session() as session:
        races = []
        horses = []
        jockeys = []
        trainers = []
        relationships = []

        for index, row in df.iterrows():
            race = {
                "race_name": f"{row['Race Date']} - {row['Race Location']} - {row['Course']} - {row['Distance']} - {row['Class']}",
                "date": row['Race Date'],
                "location": row['Race Location'],
                "course": row['Course'],
                "distance": row['Distance'],
                "condition": row['Going'],
                "class": row['Class']
            }
            races.append(race)

            horse = {"horse_name": row['Horse']}
            horses.append(horse)

            jockey = {"jockey_name": row['Jockey']}
            jockeys.append(jockey)

            trainer = {"trainer_name": row['Trainer']}
            trainers.append(trainer)

            relationship = {
                "race_name": race["race_name"],
                "horse_name": row['Horse'],
                "jockey_name": row['Jockey'],
                "trainer_name": row['Trainer'],
                "placement": row['Pla.'],
                "draw": row['Dr.'],
                "horse_weight": row['Declar. Horse Wt.']
            }
            relationships.append(relationship)

            if len(races) >= batch_size:
                session.write_transaction(create_race, races)
                session.write_transaction(create_horse, horses)
                session.write_transaction(create_jockey, jockeys)
                session.write_transaction(create_trainer, trainers)
                session.write_transaction(create_race_result_relationship, relationships)
                races = []
                horses = []
                jockeys = []
                trainers = []
                relationships = []

        if races:
            session.write_transaction(create_race, races)
            session.write_transaction(create_horse, horses)
            session.write_transaction(create_jockey, jockeys)
            session.write_transaction(create_trainer, trainers)
            session.write_transaction(create_race_result_relationship, relationships)


# Load the data into Neo4j
load_data_to_neo4j(df, batch_size=100)
