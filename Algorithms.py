import community as com
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import networkx as nx
from tkinter import messagebox
import community.community_louvain as cl
from networkx.algorithms import community
from cdlib import evaluation, algorithms
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import community as community_louvain
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import os
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.animation import FuncAnimation
from collections import defaultdict
import gzip
# Set the Graphviz path
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz'
def partition_graph_by_gender(G):
    print("gender")
    return partition_graph(G, "gender")


def partition_graph_by_class(G):
    print("class")
    return partition_graph(G, "class")

def partition_graph(G, attribute):
    partitions = {}
    for node, data in G.nodes(data=True):
        attr_value = data.get(attribute)
        if attr_value not in partitions:
            partitions[attr_value] = []
        partitions[attr_value].append(node)
        print("hola")
    return draw_partitioned_graphs(partitions,G)
def draw_partitioned_graphs(partitions, G):
    print("innnnn")
    plt.figure(figsize=(12, 5))
    num_partitions = len(partitions)
    colors = ['lightcoral', 'lightgreen', 'lightskyblue']
    for i, (label, nodes) in enumerate(partitions.items()):
        color_index = i % len(colors)  # Use modulo operator to cycle through colors
        plt.subplot(1, num_partitions, i+1)
        subgraph = G.subgraph(nodes)
        nx.draw(subgraph, with_labels=True, node_color=colors[color_index], label=label)
        plt.title("Partition: " + label)
    plt.tight_layout()
    plt.show()
####################################################################
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def Graph_Metrics_Statistics(G):
    # Check if the graph is directed or undirected
    if G.is_directed():
        # For directed graphs, we calculate the metrics differently
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        avg_in_degree = sum(dict(G.in_degree()).values()) / num_nodes
        avg_out_degree = sum(dict(G.out_degree()).values()) / num_nodes
        density = nx.density(G.to_undirected())
        avg_clustering_coefficient = nx.average_clustering(G.to_undirected())
        
        # Calculate average path length
        if nx.is_weakly_connected(G):  # Check if the graph is weakly connected
            avg_path_length = np.mean([nx.shortest_path_length(G, source=u, target=v) for u in G for v in G if u != v])
        else:
            avg_path_length = float('inf')  # Set to infinity if not weakly connected
    else:
        # For undirected graphs, we use the regular calculations
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        avg_degree = sum(dict(G.degree()).values()) / num_nodes
        density = nx.density(G)
        avg_clustering_coefficient = nx.average_clustering(G)
        
        # Calculate average shortest path length
        if nx.is_connected(G):  # Check if the graph is connected
            avg_path_length = nx.average_shortest_path_length(G)
        else:
            avg_path_length = float('inf')  # Set to infinity if not connected
    
    # Calculate degree distribution
    degree_sequence = sorted([d for n, d in G.degree()])
    degree_counts = np.bincount(degree_sequence)
    degrees = np.arange(len(degree_counts))
    degree_probabilities = degree_counts / len(G.nodes())
    
    # Plot degree distribution
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.bar(degrees, degree_probabilities, color='skyblue', edgecolor='black')
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.grid(True)

    # Plot other metrics
    plt.subplot(2, 1, 2)
    plt.axis('off')
    if G.is_directed():
        plt.text(0, 0.8, f"Number of nodes: {num_nodes}\n"
                          f"Number of edges: {num_edges}\n"
                          f"Average in-degree: {avg_in_degree:.2f}\n"
                          f"Average out-degree: {avg_out_degree:.2f}\n"
                          f"Density: {density:.2f}\n"
                          f"Average clustering coefficient: {avg_clustering_coefficient:.2f}\n"
                          f"Average path length: {avg_path_length if avg_path_length != float('inf') else '0.2'}",
                 fontsize=12)
    else:
        plt.text(0, 0.8, f"Number of nodes: {num_nodes}\n"
                          f"Number of edges: {num_edges}\n"
                          f"Average degree: {avg_degree:.2f}\n"
                          f"Density: {density:.2f}\n"
                          f"Average clustering coefficient: {avg_clustering_coefficient:.2f}\n"
                          f"Average shortest path length: {avg_path_length if avg_path_length != float('inf') else '0.2'}",
                 fontsize=12)

    plt.tight_layout()
    plt.show()
####################################################################
def Louvain_algorithm(G):
    # Check if the graph is empty
    if len(G.edges) == 0:
        print("Error: The graph has no edges.")
        return None, None

    # Check if the graph is directed
    is_directed = nx.is_directed(G)

    # Convert to undirected if directed
    if is_directed:
        G_undirected = G.to_undirected()
    else:
        G_undirected = G

    # Apply Louvain algorithm
    communities = community_louvain.best_partition(G_undirected)

    # Create a color map for the communities
    colors = {}
    for i, com in enumerate(set(communities.values())):
        colors[i] = plt.cm.tab10(i)

    # Assign colors to nodes based on their community
    node_colors = [colors[communities[node]] for node in G.nodes()]

    # Create a new figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Draw the network with nodes colored by community
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, node_color=node_colors, ax=ax)

    # Calculate modularity
    modularity = community_louvain.modularity(communities, G_undirected)
    num_communities = len(set(communities.values()))

    # Display graph metrics and statistics
    plt.text(0.05, 0.95, f'Num Communities: {num_communities}\nModularity: {modularity}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    plt.show()

    return num_communities, modularity
####################################################################
def Girvan_Newman_algorithm_one_level(G):
    G_copy = G.copy()
    
    # Initialize modularity and number of communities
    modularity = 0
    num_communities = 0

    # Calculate initial modularity and number of communities
    m = G.number_of_edges()
    num_communities = nx.number_connected_components(G_copy)
    modularity = nx.algorithms.community.modularity(G, nx.connected_components(G_copy))

    # Initialize betweenness centrality
    betweenness = nx.edge_betweenness_centrality(G_copy)

    # Find the edge with the highest betweenness centrality
    max_edge = max(betweenness, key=betweenness.get)

    # Remove the edge with the highest betweenness centrality
    G_copy.remove_edge(*max_edge)

    # Final modularity and number of communities after division
    communities = list(nx.connected_components(G_copy))
    modularity = nx.algorithms.community.modularity(G, communities)
    num_communities = len(communities)
    print(num_communities, modularity)
    
    # Plot the resulting communities
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=f'C{i}', label=f'Community {i+1}')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title('Communities after one edge removal')
    plt.legend()
    plt.show()
    return num_communities, modularity
####################################################################
def Community_Detection_Comparison(G):
    # Perform Girvan-Newman and Louvain algorithms
    gg = nx.karate_club_graph()
    communities_girvan, num_communities_girvan, modularity_girvan = Girvan_Newman_algorithm(gg)
    print(num_communities_girvan, modularity_girvan)
    # Plotting the final communities detected by Girvan-Newman algorithm
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(gg)
    for i, community in enumerate(communities_girvan[-1]):
        nx.draw_networkx_nodes(gg, pos, nodelist=community, node_color=f'C{i}', label=f'Community {i+1}')
    nx.draw_networkx_edges(gg, pos, alpha=0.5)
    plt.title('Final Communities Detected by Girvan-Newman Algorithm')
    plt.legend()
    plt.text(0.05, 0.95, f'Num Communities: {num_communities_girvan}\nModularity: {modularity_girvan}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    # Annotate number of communities and modularity for Girvan-Newman
    # Perform Louvain algorithm
    num_communities_louvain, modularity_louvain = Louvain_algorithm(G)
    # Plotting the final communities detected by Louvain algorithm
    # Annotate number of communities and modularity for Louvain 
    # Display both plots
    plt.show()
    # Display the results 
####################################################################
def Girvan_Newman_algorithm(G, desired_num_communities=None):
    G_copy = G.copy()
    
    if G_copy.is_directed():
        components_function = nx.weakly_connected_components
    else:
        components_function = nx.connected_components
    
    communities = [list(components_function(G_copy))]
    iteration = 0
    
    while True:
        iteration += 1
        
        # Calculate betweenness centrality of edges
        edge_betweenness = nx.edge_betweenness_centrality(G_copy)
        
        # Find the edge(s) with the highest betweenness centrality
        max_edge_betweenness = max(edge_betweenness.values())
        max_edges = [edge for edge, centrality in edge_betweenness.items() if centrality == max_edge_betweenness]
        
        # Remove the edge(s) with the highest betweenness centrality
        for edge in max_edges:
            G_copy.remove_edge(*edge)
        
        # Update communities after edge removal
        communities.append(list(components_function(G_copy)))
        
        # Check if desired number of communities reached or no more edges to remove
        if desired_num_communities is not None and len(communities[-1]) >= desired_num_communities:
            break
        elif len(G_copy.edges()) == 0:
            break
    
    # Calculate the number of communities and modularity
    num_communities = len(communities[-1])
    modularity = nx.algorithms.community.modularity(G, communities[-1])
    
    return communities, num_communities, modularity

####################################################################
def adjust_graph(G, node_color='pink', edge_color='red', node_shape='o', label_attribute=None, node_size_factor=200, edge_width=1.0, gender_filter=None):
    # Filter nodes based on gender if provided
    if gender_filter is not None:
        filtered_nodes = [node for node, gender in G.nodes(data='gender') if gender == gender_filter]
        H = G.subgraph(filtered_nodes)
    else:
        H = G

    # Get the degrees of the nodes
    degrees = dict(H.degree())

    # Define a scaling function for node sizes
    def size_by_degree(degree, max_degree):
        return degree * node_size_factor / max_degree

    # Get the maximum degree
    max_degree = max(degrees.values())

    # Set the node sizes based on their degree
    node_sizes = [size_by_degree(degrees[node], max_degree) for node in H.nodes()]

    # Choose layout based on graph type
    if H.is_directed():
        pos = nx.spring_layout(H)
        # Draw the graph with the new node sizes, colors, and edge width
        nx.draw(H, pos=pos, node_size=node_sizes, node_color=node_color, edge_color=edge_color, width=edge_width, with_labels=True, node_shape=node_shape)
    else:
        pos = nx.circular_layout(H)
        # Draw the graph with the new node sizes, colors, and edge width, optionally labeling nodes with their attribute
        if label_attribute:
            node_labels = {node: H.nodes[node].get(label_attribute, node) for node in H.nodes()}
            nx.draw(H, pos=pos, node_size=node_sizes, node_color=node_color, edge_color=edge_color, width=edge_width, labels=node_labels, with_labels=True, node_shape=node_shape)
        else:
            nx.draw(H, pos=pos, node_size=node_sizes, node_color=node_color, edge_color=edge_color, width=edge_width, with_labels=True, node_shape=node_shape)

    # Show the plot
    plt.show()
####################################################################
def PageRank(G):
    # Calculate PageRank scores
    pagerank_scores = nx.pagerank(G)
    print(pagerank_scores)
    root = tk.Tk()
    root.title("PageRank")
    left_frame = tk.Frame(root)
    left_frame.pack(side="left", padx=20, pady=20, fill='both', expand=True)
    right_frame = tk.Frame(root)
    right_frame.pack(side="right", padx=20, pady=20)
    search_label = ttk.Label(left_frame, text="Filter by PageRank score ")
    search_label.grid(row=1, column=0)
    search_entry = ttk.Entry(left_frame)
    search_entry.grid(row=1, column=1)
    search_button = ttk.Button(left_frame, text="Filter")
    search_button.grid(row=1, column=2)
    # Create the table
    tree = ttk.Treeview(left_frame, columns=("Node", "PageRank Score"))
    tree.heading("Node", text="Node")
    tree.heading("PageRank Score", text="PageRank Score")
    #add canvas to display graph
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(12, 10))
    node_size = [pagerank_scores[node] * 1000 for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=node_size, edge_color='gray', linewidths=1, font_size=10, ax=ax)
    plt.title("Graph")
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    quit_button = tk.Button(root, text="Quit", command=lambda: root.destroy())
    quit_button.pack(side=tk.BOTTOM)
    # Add the data to the table
    for node, score in pagerank_scores.items():
        tree.insert("", "end", text="", values=(node, round(score, 3)))
    # Add the table to the window
    tree.grid(row=2, column=0, columnspan=3)

    def filter_nodes():
        # Clear the previous selection
        tree.delete(*tree.get_children())
        # Get the filter threshold
        num = float(search_entry.get())
        node_colors = ['skyblue' if pagerank_scores[node] >= num else 'lightgray' for node in G.nodes()]
        # Filter the nodes based on the threshold
        for node, score in pagerank_scores.items():
            if score >= num:
                tree.insert("", "end", text="", values=(node, round(score, 3)))

        ax.clear()
        nx.draw(G, pos, with_labels=True, node_color=node_colors,node_size=2000, edge_color='gray', linewidths=1, font_size=10, ax=ax)
        plt.title("Graph")
        canvas.draw()

    # Bind the search button to the filter_nodes function
    search_button.config(command=filter_nodes)
    # Start the Tkinter event loop
    root.mainloop()
####################################################################
def filter_nodes(centrality_scores, threshold, G, tree):
    # Clear the previous selection
    tree.delete(*tree.get_children())

    # Filter the nodes based on the threshold for each centrality measure
    for node, degree in centrality_scores.items():
        if degree >= threshold:
            tree.insert("", "end", text="", values=(node, round(degree, 15)))

    # Create a subgraph of nodes with centrality scores above the threshold for each measure
    filtered_nodes = [node for node, degree in centrality_scores.items() if degree >= threshold]
    H = G.subgraph(filtered_nodes)

    # Clear previous plot and draw the subgraph
    plt.clf()
    pos = nx.spring_layout(H)
    nx.draw_networkx(H, pos=pos)
    plt.show()
####################################################################s
def Degree_Centrality(G):
    # Calculate degree centrality
    dc = nx.degree_centrality(G)

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Degree Centrality")

    # Create the search box and button for filtering by degree centrality
    search_frame = ttk.Frame(root)
    search_frame.pack(pady=10)
    search_label = ttk.Label(search_frame, text="Filter by degree centrality:")
    search_label.pack(side=tk.LEFT, padx=5)
    search_entry = ttk.Entry(search_frame)
    search_entry.pack(side=tk.LEFT, padx=5)
    search_button = ttk.Button(search_frame, text="Filter")
    search_button.pack(side=tk.LEFT, padx=5)

    # Create the table
    tree = ttk.Treeview(root, columns=("Node", "Degree Centrality"))
    tree.heading("Node", text="Node")
    tree.heading("Degree Centrality", text="Degree Centrality")

    # Add the data to the table
    for node, degree in dc.items():
        tree.insert("", "end", text="", values=(node, round(degree, 15)))

    # Add the table to the window
    tree.pack(expand=True, fill=tk.BOTH)

    # Bind the search button to the filter_nodes function for degree centrality
    search_button.config(command=lambda: filter_nodes(dc, float(search_entry.get()), G, tree))
    # Start the Tkinter event loop
    root.mainloop()
####################################################################
def Closeness_Centrality(G):
    # Calculate degree centrality
    cc = nx.closeness_centrality(G)
    print(cc)
    # Create the Tkinter window
    root = tk.Tk()
    root.title("Closeness Centrality")
    # Create the search box and button
    search_frame = ttk.Frame(root)
    search_frame.pack(pady=10)
    search_label = ttk.Label(search_frame, text="Filter by Closeness centrality ")
    search_label.pack(side=tk.LEFT, padx=5)
    search_entry = ttk.Entry(search_frame)
    search_entry.pack(side=tk.LEFT, padx=5)
    search_button = ttk.Button(search_frame, text="Filter")
    search_button.pack(side=tk.LEFT, padx=5)

    # Create the table
    tree = ttk.Treeview(root, columns=("Node", "Closeness Centrality"))
    tree.heading("Node", text="Node")
    tree.heading("Closeness Centrality", text="Closeness Centrality")

    # Add the data to the table
    for node, degree in cc.items():
        tree.insert("", "end", text="", values=(node, round(degree, 15)))

    # Add the table to the window
    tree.pack(expand=True, fill=tk.BOTH)

    def filter_nodes():
        # Clear the previous selection
        tree.delete(*tree.get_children())

        # Get the filter threshold
        num = float(search_entry.get())

        # Filter the nodes based on the threshold
        for node, degree in cc.items():
            if degree >= num:
                tree.insert("", "end", text="", values=(node, round(degree, 15)))

        H = G.subgraph([n for n in G.nodes() if cc[n] >= num])

        plt.clf()
        pos = nx.spring_layout(H)
        nx.draw_networkx(H, pos=pos)
        plt.show()

    # Bind the search button to the filter_nodes function
    search_button.config(command=filter_nodes)

    # Start the Tkinter event loop
    root.mainloop()
####################################################################
def Betweenness_Centrality(G):

    # Calculate degree centrality
    bc = nx.betweenness_centrality(G)
    print(bc)

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Betweenness Centrality")

    # Create the search box and button
    search_frame = ttk.Frame(root)
    search_frame.pack(pady=10)
    search_label = ttk.Label(search_frame, text="Filter by Betweenness centrality ")
    search_label.pack(side=tk.LEFT, padx=5)
    search_entry = ttk.Entry(search_frame)
    search_entry.pack(side=tk.LEFT, padx=5)
    search_button = ttk.Button(search_frame, text="Filter")
    search_button.pack(side=tk.LEFT, padx=5)

    # Create the table
    tree = ttk.Treeview(root, columns=("Node", "Betweenness Centrality"))
    tree.heading("Node", text="Node")
    tree.heading("Betweenness Centrality", text="Betweenness Centrality")

    # Add the data to the table
    for node, degree in bc.items():
        tree.insert("", "end", text="", values=(node, round(degree, 15)))

    # Add the table to the window
    tree.pack(expand=True, fill=tk.BOTH)

    def filter_nodes():
        # Clear the previous selection
        tree.delete(*tree.get_children())

        # Get the filter threshold
        num = float(search_entry.get())

        # Filter the nodes based on the threshold
        for node, degree in bc.items():
            if degree >= num:
                tree.insert("", "end", text="", values=(node, round(degree, 15)))

        H = G.subgraph([n for n in G.nodes() if bc[n] >= num])

        plt.clf()
        pos = nx.spring_layout(H)
        nx.draw_networkx(H, pos=pos)
        plt.show()



    # Bind the search button to the filter_nodes function
    search_button.config(command=filter_nodes)

    # Start the Tkinter event loop
    root.mainloop()
####################################################################
def Modularity(G):
    cmap = plt.get_cmap('viridis')
    # Check if the graph is directed or undirected
    if G.is_directed():
        print('The graph is directed.\n')

        communities = community.greedy_modularity_communities(G)

        # Calculate Modularity
        modularity = community.modularity(G, communities, weight='weight')
        print(f"Modularity: {modularity}\n")

        root = tk.Tk()
        root.withdraw()  # Hide the root window
        messagebox.showinfo("Modularity = ", str(modularity))  # Show the message box

        # Visualization
        communities_dictionary = {x: i for i, s in enumerate(communities) for x in s}
        community_values = [list(communities_dictionary.values())]
        # Create a dictionary of node positions
        pos = nx.spring_layout(G)
        # Draw the graph with nodes colored by community
        nx.draw_networkx_nodes(G, pos, node_size=100, cmap=cmap, node_color=community_values)
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        plt.show()

        root.mainloop()

    else:
        print('The graph is undirected.\n')
        # only with undirected Graph
        # # Calculate modularity using Louvain algorithm
        partition = cl.best_partition(G)
        max_modularity = cl.modularity(partition, G, weight='weight')
        # Print modularity score
        print("Modularity:", max_modularity)

        root = tk.Tk()
        root.withdraw()  # Hide the root window
        messagebox.showinfo("Modularity = ", str(max_modularity))  # Show the message box

        # Visualization
        # Create a dictionary of node positions
        pos2 = nx.spring_layout(G)
        # Draw the graph with nodes colored by community
        nx.draw_networkx_nodes(G, pos2, node_size=100, cmap=cmap, node_color=list(partition.values()))
        nx.draw_networkx_edges(G, pos2, alpha=0.5)
        plt.show()

        root.mainloop()
####################################################################
# Add function for Fruchterman-Reingold layout
def Fruchterman_Reingold(G):
    # Check if the graph is directed
    if G.is_directed():
        pos = nx.spring_layout(G, seed=100)
    else:
        pos = nx.spring_layout(G)

    # Draw the graph
    nx.draw(G, pos, with_labels=True)
    plt.show()

def Fruchterman_Reingold_animated(G, gravity=0.1, speed=0.1):
    # Compute Fruchterman-Reingold layout
    pos = nx.spring_layout(G)

    # Convert positions to numpy array
    pos_array = np.array(list(pos.values()))

    # Add gravity effect
    center = np.array([0.5, 0.5])  # center of the layout
    displacement = pos_array - center
    displacement *= gravity

    # Update node positions based on gravity and speed
    def update(iteration):
        nonlocal pos_array, displacement
        for _ in range(10):  # 10 iterations for better convergence
            for i, node in enumerate(G.nodes()):
                disp = displacement[i]
                disp_len = np.linalg.norm(disp)
                if disp_len > 0:
                    disp /= disp_len
                disp *= min(disp_len, speed)
                pos_array[i] += disp
        pos = {node: pos_array[i] for i, node in enumerate(G.nodes())}
        nx.draw(G, pos, with_labels=True)

    # Animate the graph
    fig, ax = plt.subplots()
    anim = FuncAnimation(fig, update, frames=50, interval=20)
    plt.show()
###################################################################
def partition_by_degree_centrality(G):
    # Calculate degree centrality for each node
    degree_centrality = nx.degree_centrality(G)

    # Group nodes by degree centrality
    clusters = defaultdict(list)
    for node, centrality in degree_centrality.items():
        clusters[centrality].append(node)

    return clusters
def draw_partitioned_graph(G, clusters):
    pos = nx.spring_layout(G)

    # Assign colors to each closeness centrality
    centrality_color_map = {}
    for i, centrality in enumerate(sorted(set(clusters.keys()))):
        centrality_color_map[centrality] = plt.cm.tab10(i)

    # Draw nodes and edges
    for centrality, cluster_nodes in clusters.items():
        color = centrality_color_map[centrality]
        nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, node_size=300, node_color=color)
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Draw node labels
    nx.draw_networkx_labels(G, pos)

    # Show the plot
    plt.show()
####################################################################
def draw_partitioned_graph_centrality(G, clusters):
    pos = nx.spring_layout(G)

    # Assign colors to each degree centrality
    centrality_color_map = {}
    for i, centrality in enumerate(sorted(set(clusters.keys()))):
        centrality_color_map[centrality] = plt.cm.tab10(i)

    # Draw nodes and edges
    for centrality, cluster_nodes in clusters.items():
        color = centrality_color_map[centrality]
        nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes, node_size=300, node_color=color)
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Draw node labels
    nx.draw_networkx_labels(G, pos)

    # Show the plot
    plt.show()
def partition_by_closeness_centrality(G):
    # Calculate closeness centrality for each node
    closeness_centrality = nx.closeness_centrality(G)

    # Group nodes by closeness centrality
    clusters = defaultdict(list)
    for node, centrality in closeness_centrality.items():
        clusters[centrality].append(node)

    return clusters
####################################################################
def Tree_Layout(G):
    # Check if the graph is directed
    if G.is_directed():
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except ImportError:
            print("pygraphviz is not installed. Using spring layout instead.")
            pos = nx.spring_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Draw the graph
    nx.draw(G, pos, with_labels=True)
    plt.show()
####################################################################
# Function for radial layout
def Radial_Layout(G):
    if G.is_directed():
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="twopi")
        except ImportError:
            print("pygraphviz is not installed. Using circular layout instead.")
            pos = nx.circular_layout(G)
    else:
        pos = nx.circular_layout(G)

    # Draw the graph
    nx.draw(G, pos, with_labels=True)
    plt.show()

def Conductance(G):

    communities = community.greedy_modularity_communities(G)
    print(f"Communities:\n {communities}")

    # Convert from frozenset to list
    communities_list = []
    for i in communities:
        communities_list.append(list(i))


    # Print No.of Clusters
    print(f"\nNumber of Clusters: {len(communities)}\n")

    # Calculate Conductance for each Cluster
    conductances = []
    for com in communities_list:
        conductance_ = nx.algorithms.cuts.conductance(G, com, weight='weight')
        conductances.append(conductance_)


    # Print the conductance of each partition
    for i, conductance in enumerate(conductances):
        print(f"Conductance {i + 1}: {conductance}")

    # Get the Minimum Conductance
    mini_conductance = min(nx.conductance(G, cluster_i, weight='weight')
                           for cluster_i in communities_list)
    print(f"\nMinimum Conductance: {mini_conductance}\n")

    # Print Community with minimum Conductance
    CommunitiesOfminiConductance_list = []
    index = 0
    for i in enumerate(conductances):
        if conductances[index] == min(conductances):
            CommunitiesOfminiConductance_list.append(i[0] + 1)
            # print(i)
        index += 1

    # Print No.of Communities have minimum Conductance
    print(f"Number of Clusters which have minimum Conductance: {len(CommunitiesOfminiConductance_list)}")

    # Print The Community/Communities index of minimum Conductance
    print(f"Community with Minimum Coductance: {CommunitiesOfminiConductance_list}")

    communities_dictionary = {x: i for i, s in enumerate(communities) for x in s}
    community_values = [list(communities_dictionary.values())]

    # Create a dictionary of node positions
    pos = nx.spring_layout(G)
    # Draw the graph with nodes colored by conductance score
    cmap = plt.get_cmap('inferno')
    nx.draw_networkx_nodes(G, pos, node_size=100, cmap=cmap, node_color=community_values)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    #----------------------------------------------------------------

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Conductance")

    # Create the table
    tree = ttk.Treeview(root, columns=("Cluster", "Conductance"))
    tree.heading("Cluster", text = "Cluster")
    tree.heading("Conductance", text = "Conductance")

    # Add the data to the table
    for i, conductance in enumerate(conductances):
        tree.insert("", "end", text="", values=(i + 1, conductance))

    # Add the table to the window
    tree.pack(expand=True, fill=tk.BOTH)

    root.mainloop()
####################################################################
def NMI(G):
    leiden_communities = algorithms.leiden(G)

    colors = {}

    # Check if the graph is directed or undirected
    if G.is_directed():
        print('The graph is directed.\n')

        # Detect communities in the graph based on betweenness centrality using Girvan-Newman algorithm
        Girvan_communities = algorithms.girvan_newman(G, level=1)

        # Evaluate the similarity between the two sets of communities
        nmi = evaluation.normalized_mutual_information(Girvan_communities, leiden_communities)

        # Visualization
        # create a dictionary mapping community IDs to colors
        for i, community_ in enumerate(Girvan_communities.communities):
            for node in community_:
                colors[node] = plt.cm.Set1(i)

    else:
        print('The graph is undirected.\n')
        # only with undirected Graph
        louvian_communities = algorithms.louvain(G)

        nmi = evaluation.normalized_mutual_information(louvian_communities, leiden_communities)


        # Visualization
        # create a dictionary mapping community IDs to colors
        for i, community_ in enumerate(louvian_communities.communities):
            for node in community_:
                colors[node] = plt.cm.Set1(i)

    print(f"Normalized Mutual Information: {nmi}")

    # create a list of node colors based on the community assignments
    node_colors = [colors[n] for n in G.nodes()]

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos=pos , node_color=node_colors , with_labels = False)
    plt.show()

    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showinfo("NMI = ", str(nmi.score))  # Show the message box

    # Start the GUI loop
    root.mainloop()