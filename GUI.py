import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from tkinter import filedialog
import Algorithms
import GUI
import networkx as nx
import preprocessing

G = nx.DiGraph()

def homePage():
    root = tk.Tk()
    root.geometry("500x300")
    root.title("Social Network analysis")
    label = tk.Label(root, text="Home Page", font=12)
    label.pack()

    def radio_click1():
        GUI.G = nx.Graph()

    def radio_click2():
        GUI.G = nx.DiGraph()

    # Create the radio buttons
    option1 = tk.Radiobutton(root, text="Undirected", value="1", command=radio_click1)
    option2 = tk.Radiobutton(root, text="Directed", value="2", command=radio_click2)

    # Pack the radio buttons
    option1.pack(pady=5)
    option2.pack(pady=10)

    def button1_click():
        # Show a file dialog window and get the selected file path
        node_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        preprocessing.readNodes(G, node_path)

    def button2_click():
        # Show a file dialog window and get the selected file path
        edges_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        preprocessing.readEdges(G, edges_path)

    def Next():
        # Navigate to algorithms page
        print("Clicked")
        select_algorithm(root)

    button1 = tk.Button(root, text="Browse Nodes", command=button1_click)
    button1.pack(pady=5)

    button2 = tk.Button(root, text="Browse Edges", command=button2_click)
    button2.pack(pady=5)

    button3 = tk.Button(root, text="Next", command=Next)
    button3.pack(pady=5)

    root.mainloop()

def select_algorithm(root):
    ty = "Directed" if G.is_directed() else "Undirected"
    print("Type of the graph is " + ty)
    # Create a new top-level window
    new_window = tk.Toplevel(root)
    new_window.geometry("500x300")
    new_window.title("Techniques")

    label = tk.Label(new_window, text="Select an option", font=12)
    label.pack()

    # Create a list of options for the combobox
    options = ["Louvain algorithm", "Modularity", "Conductance",
               "NMI", "Page rank", "Degree centrality", "Closeness centrality","Betweenness centrality", "Adjust graph", "Fruchterman Reingold", "Radial Layout", "Tree Layout", "Girvan_Newman_one_level","Girvan_Newman_all_level","Community Detection Comparison","Graph Metrics and Statistics","Gender","class","Fruchterman Reingold animated","Degree filter","partitioning by_d","partitioning by_c"]

    # Create a StringVar object to store the selected option
    selected_option = tk.StringVar()

    # Create the combobox widget and pack it onto the window
    combobox = ttk.Combobox(new_window, values=options, textvariable=selected_option)
    combobox.pack()

    # Create a function to handle combobox selection events
    def combobox_selected(event):
        if selected_option.get() == options[0]:
            Algorithms.Louvain_algorithm(G)
        elif selected_option.get() == options[8]:
            # Algorithms.adjust_graph(G, node_color='blue', edge_color='black', node_shape='^', label_attribute='class', node_size_factor=500,)
            param_window = tk.Toplevel(new_window)
            param_window.geometry("300x200")
            param_window.title("Adjust Graph Parameters")

            # Create labels and input fields for each parameter
            node_color_label = tk.Label(param_window, text="Node Color:")
            node_color_label.pack()
            node_color_entry = tk.Entry(param_window)
            node_color_entry.pack()

            edge_color_label = tk.Label(param_window, text="Edge Color:")
            edge_color_label.pack()
            edge_color_entry = tk.Entry(param_window)
            edge_color_entry.pack()

            node_shape_label = tk.Label(param_window, text="Node Shape:")
            node_shape_label.pack()
            node_shape_entry = tk.Entry(param_window)
            node_shape_entry.pack()

            label_attribute_label = tk.Label(param_window, text="Label Attribute:")
            label_attribute_label.pack()
            label_attribute_entry = tk.Entry(param_window)
            label_attribute_entry.pack()

            node_size_factor_label = tk.Label(param_window, text="Node Size Factor:")
            node_size_factor_label.pack()
            node_size_factor_entry = tk.Entry(param_window)
            node_size_factor_entry.pack()
            edge_size_factor_label = tk.Label(param_window, text="Edge Size Factor:")
            edge_size_factor_label.pack()
            edge_size_factor_entry = tk.Entry(param_window)
            edge_size_factor_entry.pack()
            gender_factor_label = tk.Label(param_window, text="Edge Size Factor:")
            gender_factor_label .pack()
            gender_factor_entry = tk.Entry(param_window)
            gender_factor_entry.pack()
            def execute_adjust_graph():
                # Get the parameter values from the input fields
                node_color = node_color_entry.get()
                edge_color = edge_color_entry.get()
                node_shape = node_shape_entry.get()
                gender=gender_factor_entry.get()
                label_attribute = label_attribute_entry.get()
                node_size_factor = int(node_size_factor_entry.get())
                edge_size_factor=int(edge_size_factor_entry.get())
            
                # Call the adjust_graph function with the provided parameters
                Algorithms.adjust_graph(G, node_color, edge_color, node_shape, label_attribute, node_size_factor,edge_size_factor,gender)

                # Close the parameter window after executing the function
                param_window.destroy()

            # Create a button to execute the adjust_graph function
            execute_button = tk.Button(param_window, text="Execute", command=execute_adjust_graph)
            execute_button.pack()
        elif selected_option.get() == options[4]:
            Algorithms.PageRank(G)
        elif selected_option.get() == options[5]:
            Algorithms.Degree_Centrality(G)
        elif selected_option.get() == options[6]:
            Algorithms.Closeness_Centrality(G)
        elif selected_option.get() == options[7]:
            Algorithms.Betweenness_Centrality(G)
        elif selected_option.get() == options[1]:
            Algorithms.Modularity(G)
        elif selected_option.get() == options[2]:
            Algorithms.Conductance(G)
        elif selected_option.get() == options[3]:
            Algorithms.NMI(G)
        elif selected_option.get() == options[9]:
            Algorithms.Fruchterman_Reingold(G)
        elif selected_option.get() == options[11]:
            Algorithms.Tree_Layout(G)
        elif selected_option.get() == options[10]:
            Algorithms.Radial_Layout(G)
        elif selected_option.get() == options[12]:
            Algorithms.Girvan_Newman_algorithm_one_level(G)
        elif selected_option.get() == options[13]:
             if selected_option.get() == options[13]:
                g= nx.karate_club_graph()
                communities,num_communities, modularity = Algorithms.Girvan_Newman_algorithm(g)
                print(num_communities, modularity)
                # Plotting the final communities
                plt.figure(figsize=(10, 6))
                pos = nx.spring_layout(g)
                for i, community in enumerate(communities[-1]):
                    nx.draw_networkx_nodes(g, pos, nodelist=community, node_color=f'C{i}', label=f'Community {i+1}')
                nx.draw_networkx_edges(g, pos, alpha=0.5)
                plt.title('Final Communities Detected by Girvan-Newman Algorithm')
                plt.legend()
                plt.show()
        elif selected_option.get() == options[14]:
        # Run the community detection comparison function
              Algorithms.Community_Detection_Comparison(G)       
        elif selected_option.get()== options[15]:
            Algorithms.Graph_Metrics_Statistics(G)
        elif selected_option.get()==options[16]:
            Algorithms.partition_graph(G,"gender")
        elif selected_option.get()==options[17]:
            Algorithms.partition_graph(G,"class")
        elif selected_option.get()==options[18]:
            Algorithms.Fruchterman_Reingold_animated(G, gravity=0.1, speed=0.1)
        elif selected_option.get()==options[19]:
            param_window = tk.Toplevel(new_window)
            param_window.geometry("300x200")
            param_window.title("Adjust Graph Parameters")

            # Create labels and input fields for each parameter
            min_label = tk.Label(param_window, text="Min range:")
            min_label.pack()
            min_entry = tk.Entry(param_window)
            min_entry.pack()

            max_label = tk.Label(param_window, text="Max range")
            max_label.pack()
            max_entry = tk.Entry(param_window)
            max_entry.pack()
            def execute_filter():
                # Get the parameter values from the input fields
                min = int(min_entry.get())
                max = int(max_entry.get())
                # Call the adjust_graph function with the provided parameters
                Algorithms.filter_and_visualize_graph(G, degree_range=(min, max)) 
                # Close the parameter window after executing the function
                param_window.destroy()
            # Create a button to execute the adjust_graph function
            execute_button = tk.Button(param_window, text="Execute", command=execute_filter)
            execute_button.pack()
        elif selected_option.get()==options[20]:
            clusters = Algorithms.partition_by_degree_centrality(G)
            Algorithms.draw_partitioned_graph(G, clusters)
        elif selected_option.get()==options[21]:
            clusters = Algorithms.partition_by_closeness_centrality(G)
            Algorithms.draw_partitioned_graph_centrality(G, clusters)  
    # Bind the combobox selection event to the combobox_selected function
    combobox.bind("<<ComboboxSelected>>", combobox_selected)
    combobox.pack(pady=60)

    # Set the default option
    selected_option.set(options[0])
homePage()
