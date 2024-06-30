from py2neo import Graph
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from neo4j import GraphDatabase
import pandas as pd

# Replace with your Neo4j connection details
uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"

# Create a Neo4j driver instance
driver = GraphDatabase.driver(uri, auth=(user, password))


def query_to_dataframe(query):
    with driver.session() as session:
        # Execute the Cypher query
        result = session.run(query)

        # Convert the result to a list of dictionaries
        data = [record.data() for record in result]

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)

        return df


# Query to extract data
query = """
MATCH (h:Horse)-[p:PARTICIPATED_IN]->(r:Race)
OPTIONAL MATCH (j:Jockey)-[:RIDDEN]->(h)
OPTIONAL MATCH (t:Trainer)-[:TRAINED]->(h)
WHERE r.date > date("2024-06-01")
RETURN h, p, r, j, t
LIMIT 25
"""

# Get the result as a DataFrame
data = query_to_dataframe(query)

# Create a NetworkX graph from the data
G = nx.Graph()

# Add nodes and edges to the graph
for index, row in data.iterrows():
    G.add_node(row['h']['horse_name'], entity='horse', sex=row['h']['horse_sex'], origin=row['h']['horse_origin'], colour=row['h']['horse_colour'])
    G.add_node(row[0]['r.race_name'], entity='race', date=row[0]['r.race_date'], location=row[0]['r.race_location'], course=row[0]['r.race_course'], distance=row[0]['r.race_distance'], condition=row[0]['r.race_condition'], class_=row[0]['r.race_class'])
    G.add_node(row[0]['j.jockey_name'], entity='jockey')
    G.add_node(row[0]['t.trainer_name'], entity='trainer')
    G.add_edge(row[0]['h.horse_name'], row[0]['r.race_name'], placement=row[0]['p.race_placement'], weight=row[0]['p.horse_weight'], draw=row[0]['p.race_draw'])
    G.add_edge(row[0]['j.jockey_name'], row[0]['h.horse_name'], relation='ridden')
    G.add_edge(row[0]['t.trainer_name'], row[0]['h.horse_name'], relation='trained')

# Convert NetworkX graph to PyTorch Geometric Data object
data = Data.from_networkx(G)


class RacePlacementGNN(torch.nn.Module):
    def __init__(self):
        super(RacePlacementGNN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc1 = torch.nn.Linear(16, 8)
        self.fc2 = torch.nn.Linear(8, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

model = RacePlacementGNN()


# Create a DataLoader for batching
loader = DataLoader([data], batch_size=1, shuffle=True)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(100):
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)  # Assuming 'y' is the target race placement
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item()}")

