# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-09-08 22:24:21
#  * @modify date 2024-09-08 22:24:21
#  * @desc [description]
#  */


import pandas as pd
from GPA_01092024.graph import EdgePhaseGraph,PhaseGraph,plot_events
import networkx as nx

path = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/eq/eqscenario.csv"
df = pd.read_csv(path)

es = EdgePhaseGraph(df)
# es.add_lead_node()
output = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/graph/eqgraph.graphml"
# edges = es.get_structure(with_labels=True,output=output)


g = nx.read_graphml(output)
node_data = g.nodes(data=True)
df = pd.DataFrame.from_dict(dict(node_data), orient='index')
print(df)
# print(nx.get_node_attributes(g))

pg = PhaseGraph(g)
print(pg.node_data)
pg.asso()
print(pg.node_data)

plot_events(pg.graph,pg.node_data,order=(31.61616153307329,-103.985011684094))

# output = "/home/emmanuel/ecastillo/dev/associator/GPA/project/delaware_west/dataset/eq/edges.csv"
# edges.to_csv(output,index=False)
# print(edges)
