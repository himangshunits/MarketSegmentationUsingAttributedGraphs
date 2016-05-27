
################################################################################################
# PLEASE README!
################################################################################################

# The Code for project 06.Topic-7.Project-6.MarketSegmentation.AttributedGraphCommunityDetection
# CSC 591: Algorithms for Data Guided Business Intelligence.
# Student name : Himangshu Ranjan Borah
# Unity Id: hborah
# Student Id: 200105222
# Description: The code uses the following Libraries.
# copy, numpy, pandas, igraph, sklearn, colour
# The data folder must be present in the directory where this file resides.
# The Program generates the final clusters in a TXT file called "communities.txt" and a visualization
# of the clusters in the original graph in the file called "visual_cluster_plot_<alphavalue>.pdf"
# Please run the code as python sac1.py <alpha value> from the directory it resides.
# The MAX_EPOCH is the maximum no. of times the convergence of phase1 and phase2 runs before it converges.

# Estimated Running times:

# Alpha 1 = Phase 2 Converged Succesfully with clusters = 6
# 50.836261 seconds taken for running!

# Alpha 0.5 = Phase 2 Converged Succesfully with clusters = 3
# 945.605689 seconds taken for running!

# Alpha 0.0 = Phase 2 Converged Succesfully with clusters = 2
# 354.026016 seconds taken for running!

################################################################################################
################################################################################################

import sys
import collections
import time
import csv
import math
import copy
import numpy as np
import pandas as pd
import igraph as gp
from itertools import izip
from sklearn.metrics.pairwise import cosine_similarity
import colour

MAX_EPOCH = 15


################################################################################################
# CODE FOR VISUALIZATION OR THE CLUSTERS
################################################################################################

# Input parsing
if len(sys.argv) != 2:
	print "The input format is not proper ! Please enter in the following format."
	print "python sac1.py <alpha value>"    
	exit(1)
alpha_val_str = sys.argv[1]
alpha_value = float(alpha_val_str)


# Function to plot an igraph object.
def sac_plot(sac_graph):
	layout = sac_graph.layout("kk")
	gp.plot(sac_graph, layout=layout)


# Fucntion to plot an I graph object with different clusters having different colors according to a membership list.
def sac_plot_cluster(sac_graph, membership_list, filename):
	red = colour.Color("red")
	blue = colour.Color("blue")
	my_rainbow = list(red.range_to(blue, membership_list.max() + 1))
	color_list = [i.get_hex_l() for i in my_rainbow]
	layout = sac_graph.layout("kk")
	gp.plot(sac_graph, filename, layout=layout, vertex_color=[color_list[x] for x in membership_list])

# Same as above function, but from a cluster_dict. Saves the plot to filename.
def plot_clusters_from_dict(clusters_dict, sac_graph_original, filename):
	membership_list = [0] * len(sac_graph_original.vs)
	membership_list = pd.Series(membership_list)
	for key in clusters_dict.iterkeys():
		indices = list(clusters_dict[key])
		cluster_id = key
		for index in indices:
			membership_list[index] = cluster_id
	sac_plot_cluster(sac_graph_original, membership_list, filename)		


################################################################################################
# Main Code for Segmentation starts here.
################################################################################################
	


def main():	
	if alpha_value >= 0 and alpha_value <= 1:
		# Call the main executing fucntion.
		print "The alpha Value is = " + str(alpha_value)
		start_time = time.clock()
		sac1_clustering()
		print time.clock() - start_time, "seconds taken for running!"
	else:
		print "The alpha value is not proper! Please enter a alpha between 0 and 1 (closed) as below."
		print "python sac1.py <alpha value>"


# Code to build the Graph from the Data in the files.
def load_data_to_graph():
	# Add edges from file
	sac_graph = gp.Graph.Read_Edgelist('data/fb_caltech_small_edgelist.txt', directed=False)
	sac_attributes = pd.read_csv("data/fb_caltech_small_attrlist.csv")
	headers = sac_attributes.columns.tolist()
	# add the attributes vectors to the vertices og the graph as an attribute called "attribute_vector."
	for i, item in sac_attributes.iterrows():
		sac_graph.vs[i]["attribute_vector"] = item
	
	# Set the edge weights to 1.
	sac_graph.es["weight"] = 1
	# This attribute is used to keep track of how many original nodes of the main graph is represnted by this node.
	# For epoch 1 of phase2, it will be always 1, but when the graphs strat to contract, it's value will increase.
	# This is used to find the actual no. of community memebers while calculating similarity.
	sac_graph.vs["cardinality"] = 1

	return sac_graph


# Find the net gain in cosine similarity when moving x from it's community to the community "community_id"
# This includes the gain in similarities by moving to new community - loss incurred. 
# Details in piazza post https://piazza.com/class/ighszehbqsl7ee?cid=430
def cos_similarity(x, membership_list, community_id, sac_graph, consine_data, no_of_clusters):
	count_for_new_cluster = 0.0
	count_for_old_cluster = 0.0
	new_community = membership_list[membership_list == community_id].index
	for i in new_community:
		count_for_new_cluster += consine_data[i, x]

	community_of_x = membership_list[x]
	old_community = set(membership_list[membership_list == community_of_x].index)

	# Safety check on the communitites.
	if x not in old_community:
		print "Severe Error! Abort ! X not in his old cummunity."
		print x, community_of_x, old_community
		exit(1)

	old_community = old_community - set([x])

	# Safety check on the communitites.
	if x in old_community:
		print "Severe Error! Abort ! X still in his old cummunity."
		print x, community_of_x, old_community
		exit(1)
	
	for i in old_community:
		count_for_old_cluster += consine_data[i, x]

	# Find the cardinalities as to how many nodes actually one node represents.	
	new_cardinalities = map(lambda x: sac_graph.vs[x]["cardinality"], new_community)
	old_cardinalities = map(lambda x: sac_graph.vs[x]["cardinality"], old_community)

	# We divide by square of the community sizes.
	if(len(old_cardinalities) == 0): # Division by zero error.
		return float(count_for_new_cluster)/(sum(new_cardinalities)**2)
	else:
		return float(count_for_new_cluster)/(sum(new_cardinalities)**2) - float(count_for_old_cluster)/(sum(old_cardinalities)**2)


# Functions to write down the clusters to a file called "filename"
def write_clusters_new(clusters, filename):
	f = open(filename,'w')
	line_to_write = "\n".join(",".join(str(x) for x in clusters[item]) for item in clusters.iterkeys())
	f.write(line_to_write)	
	f.close()


# Function to abstract out the inner loop for j according to the pseudo code in the paper.
def inner_loop_j(j, i, sac_graph, membership_list, consine_data, old_modularity_newman):
	temp_membership = copy.deepcopy(membership_list)
	community_of_j = membership_list[j]
	temp_membership[i] = community_of_j
	
	# Check for boundary vals for optimization
	if alpha_value == 1:
		delta_newman =  sac_graph.modularity(temp_membership, weights=sac_graph.es['weight']) - old_modularity_newman
		delta_sim = 0
	elif alpha_value == 0:
		delta_newman = 0	
		delta_sim = cos_similarity(i, membership_list, community_of_j, sac_graph, consine_data, no_of_clusters = 0)
	else:
		delta_newman =  sac_graph.modularity(temp_membership, weights=sac_graph.es['weight']) - old_modularity_newman	
		delta_sim = cos_similarity(i, membership_list, community_of_j, sac_graph, consine_data, no_of_clusters = 0)

	composite_delta = alpha_value * delta_newman + (1 - alpha_value) * delta_sim

	return composite_delta


def sac_phase1(sac_graph, membership_list, sac_graph_original, phase_2_epoch):
	no_of_nodes = len(sac_graph.vs)
	
	consine_data = np.zeros((no_of_nodes, no_of_nodes))
	# Calculate The cosine Matrix for the current graph beforehand for faster execution.
	if alpha_value != 1:
		print "Start Building Cosine Dist for current Graph"
		for i in range(no_of_nodes):
			# i,i set to zero ensures that it doesnt calculate similarity with itself
			for j in range(i, no_of_nodes):
				consine_data[i, j] = cosine_similarity(sac_graph.vs[i]["attribute_vector"].reshape(1, -1), sac_graph.vs[j]["attribute_vector"].reshape(1, -1))
				consine_data[j, i] = consine_data[i, j]
		print "End Building Cosine Dist!"		

	phase_1_epoch = MAX_EPOCH
	while phase_1_epoch > 0:
		last_community = copy.deepcopy(membership_list)
		for i in range(no_of_nodes):
			print "Phase 2 Epoch, Phase 1 Epoch, Current Node = " + str(MAX_EPOCH - phase_2_epoch + 1) + " :: " + str(MAX_EPOCH - phase_1_epoch + 1) + " :: " + str(i)
			max_comp_mod = float('-inf')
			max_composite_j = 0
			old_modularity_newman = sac_graph.modularity(membership_list, weights=sac_graph.es['weight'])
			no_of_clusters = membership_list.nunique()

			for j in membership_list.unique():				
				# Run the inner loop.
				composite_delta = inner_loop_j(j, i, sac_graph, membership_list, consine_data, old_modularity_newman)
				if composite_delta > max_comp_mod:
					max_comp_mod = composite_delta
					max_composite_j = j

			# Check if positive max_comp_mod
			if max_comp_mod > 0:
				# Change the original membership_list. This is the point where nodes are moving across clusters.
				#print "POSITIVE MOD = " + str(max_comp_mod)
				membership_list[i] = membership_list[max_composite_j]


		if last_community.equals(membership_list):
			print "Phase 1 Converged !"
			break
		else:			
			phase_1_epoch = phase_1_epoch - 1	
	return membership_list		


# Merging function for the attribute_vector
def atrribute_mean(x):
	return sum(x)/len(x)

def atrribute_sum(x):
	return sum(x)	


def sac1_clustering():
	# Get the graph
	sac_graph = load_data_to_graph()

	# Get a permemnat copy for the attribute preservation
	sac_graph_original = sac_graph.copy()


	# Define the inital clusters dict
	clusters_dict = dict()	
	for item in range(len(sac_graph.vs)):
		clusters_dict[item] = [item]

	# Intial membership list
	membership_list = pd.Series(range(len(sac_graph.vs)))
	
	# Phase 2 loop starts here
	phase_2_epoch = MAX_EPOCH

	while(phase_2_epoch > 0):
		#print sac_graph
		# Intialize to separate communities:
		no_of_nodes = len(sac_graph.vs)
		# Check the cases when it merges to one cluster!
		if no_of_nodes == 1:
			print "Alert All Merged! 1 vertices after iteration = " + str(phase_2_epoch)
			break

		last_community = copy.deepcopy(membership_list)
		membership_list = pd.Series(range(no_of_nodes))

		membership_list = sac_phase1(sac_graph, membership_list, sac_graph_original, phase_2_epoch)

		unique_clusters = membership_list.unique()		
		panda_unique = pd.Series(unique_clusters)
		# map the clusters to new cluster IDs
		mapped_membership = membership_list.map(lambda x: pd.Index(panda_unique).get_loc(x))

		# The below is needed after the phase 2 is complete
		next_clusters_dict = dict()

		#print clusters_dict
		
		for item in range(len(unique_clusters)):
			curr_cluster = list(mapped_membership[mapped_membership == item].index)
			# Now, the next_clusters_dict will have expanded contents from the last clusters_dict
			next_cluster = list()
			for item_c in curr_cluster:
				next_cluster = next_cluster + clusters_dict[item_c]
			next_clusters_dict[item] = next_cluster
		
		#print clusters_dict
		clusters_dict = next_clusters_dict	

		sac_graph.contract_vertices(mapped_membership, combine_attrs=dict(cardinality=sum, attribute_vector=atrribute_mean))
		sac_graph.simplify(loops=False, combine_edges=sum)
		# Removing loop with horrify the results!

		#print(sac_graph)
		print clusters_dict
		#sac_plot(sac_graph)
		if last_community.equals(membership_list):
			print "Phase 2 Converged Succesfully with clusters = " + str(len(clusters_dict))
			break
		else:			
			phase_2_epoch -= 1

	# end phase 2 loop

	# Write the results and plot the graphs.
	write_clusters_new(clusters_dict, "communities.txt")
	plot_clusters_from_dict(clusters_dict, sac_graph_original, filename = "visual_cluster_plot_" + str(alpha_value) + ".pdf")





# Call the main. Entry point.

if __name__ == "__main__":
	main()

