import pickle
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder
import sys
import os

final_graph = {}

pickle_in = open("./struc2vec/pickles/distances_nets_graphs.pickle", "rb")
graph_dict = pickle.load(pickle_in, encoding="bytes")

taskname = str(sys.argv[1])
#print("Graph")
for layer,subgraph in graph_dict.items():
  #print("Number of nodes in layer "+str(layer)+" is "+str(len(subgraph)))
  filename = "./struc2vec/pickles/distances_nets_weights-layer-"+str(layer)+".pickle"
  #print(filename)
  pickle_temp = open(filename, "rb")
  distance_nets_weights = pickle.load(pickle_temp, encoding="bytes")
  final_subgraph = {}
  for nodeid,neighbors in subgraph.items():
    #print(len(distance_nets_weights[nodeid]) == len(neighbors))
    final_subgraph[nodeid] = (neighbors, distance_nets_weights[nodeid])
  final_graph[layer] = final_subgraph
#print()

pickle_in = open("./struc2vec/pickles/amount_neighbours.pickle", "rb")
amount_neighbours_dict = pickle.load(pickle_in, encoding="bytes")

#print("Amount Neighbours info")
for layer,subgraph in amount_neighbours_dict.items():
  print("Number of nodes in layer "+str(layer)+" is "+str(len(subgraph)))
  num_edges_in_layer = 0
  for nodeid,down_weight in subgraph.items():
    #print("Layer "+str(layer)+": Node id "+str(nodeid)+"'s down weight is "+str(down_weight))
    tup = final_graph[layer][nodeid]
    final_subdict = {}
    final_subdict["neighbor ids"] = tup[0]
    num_edges_in_layer += len(tup[0])
    final_subdict["weights to neighbor ids"] = tup[1]
    final_subdict["weight to down layer"] = down_weight
    final_graph[layer][nodeid] = final_subdict
  #print("layer " + str(layer) + " total edges are " + str(num_edges_in_layer))
#print()


#Final graph dict is generated

count = 0

node_dict = {}
x = []
edge_index = []
edge_attr = []
edge_color = []
y = []

#taskname = "barbel_50_5"

# If node id starts from 1, this should be 1. If starts from 0, then 0
node_id_starts_from_one = 0

tmp = open(str(sys.argv[2]), 'r').readlines()[0]
old_node_labels = tmp.strip('][').split(', ')
old_node_labels = list(map(int, old_node_labels))
#print(type(old_node_labels))
#print(old_node_labels)

total_num_layers = len(final_graph)
total_num_nodes_in_original_graph = len(final_graph[0])

for layer,subgraph in final_graph.items():
  node_dict[layer] = {}
  print (layer)
  for nodeid,another_subgraph in subgraph.items():
    #print (layer, nodeid)
    node_dict[layer][nodeid] = count
    sublist = []
    sublist.append(layer)
    sublist.append(nodeid)
    x.append(sublist)
    #print(nodeid)
    new_node_label = old_node_labels[nodeid-node_id_starts_from_one]
    y.append(new_node_label)
    count = count + 1
  if (str(sys.argv[4]) == "with_extra"):
    nodeid = total_num_nodes_in_original_graph+1
    node_dict[layer][nodeid] = count
    sublist = []
    sublist.append(layer)
    sublist.append(nodeid)
    x.append(sublist)
    new_node_label = max(old_node_labels)+1
    y.append(new_node_label)
    count = count + 1

edge_index_1 = []
edge_index_2 = []


for layer,subgraph in final_graph.items():
  print (layer)
  for nodeid,another_subgraph in subgraph.items():
    neighbor_ids_list = final_graph[layer][nodeid]["neighbor ids"]
    weights_to_neighbor_ids_list = final_graph[layer][nodeid]["weights to neighbor ids"]
    if (str(sys.argv[4]) == "with_extra"):
      first = node_dict[layer][nodeid]
      second = node_dict[layer][total_num_nodes_in_original_graph+1]
      edge_index_1.append(first)
      edge_index_2.append(second)
      ele = []
      dummy_weight_info = 1
      ele.append(dummy_weight_info)
      edge_attr.append(ele)
      edge_color.append(layer)
      edge_index_1.append(second)
      edge_index_2.append(first)
      ele = []
      dummy_weight_info = 1
      ele.append(dummy_weight_info)
      edge_attr.append(ele)
      edge_color.append(layer)
    for idx in range(len(neighbor_ids_list)):
      first = node_dict[layer][nodeid]
      second = node_dict[layer][neighbor_ids_list[idx]]
      edge_index_1.append(first)
      edge_index_2.append(second)
      ele = []
      ele.append(weights_to_neighbor_ids_list[idx])
      edge_attr.append(ele)
      edge_color.append(layer)
    if(layer != total_num_layers-1):
      first = node_dict[layer][nodeid]
      if nodeid in node_dict[layer+1]:
        second = node_dict[layer+1][nodeid]
        edge_index_1.append(first)
        edge_index_2.append(second)
        ele = []
        down_edge_weight_info = final_graph[layer][nodeid]["weight to down layer"]
        down_edge_weight_info = math.log(down_edge_weight_info + math.e)
        ele.append(down_edge_weight_info)
        edge_attr.append(ele)
        edge_color.append(total_num_layers)
        edge_index_1.append(second)
        edge_index_2.append(first)
        ele = []
        up_edge_weight_info = 1
        ele.append(up_edge_weight_info)
        edge_attr.append(ele)
        edge_color.append(total_num_layers)

edge_index.append(edge_index_1)
edge_index.append(edge_index_2)

edge_index = np.array(edge_index)
edge_attr = np.array(edge_attr)
edge_color = np.array(edge_color)
x = np.array(x)
y = np.array(y)

#One hot encoding
#print(x)
onehotencoder = OneHotEncoder(categories = 'auto')
x = onehotencoder.fit_transform(x).toarray()

mapper = {}
for layer,subgraph in node_dict.items():
  for oldnodeid,newnodeid in subgraph.items():
    if (oldnodeid == total_num_nodes_in_original_graph+1):
      continue
    if oldnodeid in mapper:
      mapper[oldnodeid][0].append(newnodeid)
    else:
      mapper[oldnodeid] = ([newnodeid], old_node_labels[oldnodeid-node_id_starts_from_one])

#print(x)
print(x.shape)
#print(edge_index)
print(edge_index.shape)
#print(edge_attr)
print(edge_attr.shape)
#print(edge_color)
print(edge_color.shape)
#print(y)
print(y.shape)
#print(mapper)
print(len(mapper))

aggregated_data = []
aggregated_data.append(x.shape)
aggregated_data.append(edge_index)
aggregated_data.append(edge_attr)
aggregated_data.append(y)
aggregated_data.append(mapper)
aggregated_data.append(edge_color)
folder_path = "multilayer_datasets/" + str(sys.argv[3]) + "_dataset/"  + str(sys.argv[3]) + "_" + str(sys.argv[4]) + "_" + str(sys.argv[5]) + "_pickles/"
try:
  os.makedirs(folder_path)
except:
  pass
print("PICKLING NOW")
pickle.dump(aggregated_data, open(folder_path + taskname + ".pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
