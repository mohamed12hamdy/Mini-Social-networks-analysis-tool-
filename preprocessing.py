import csv

map2 = {}

def readNodes(G, path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            node_id = row[0]
            # node_class = row[1]
            node_gender = row[1]
            G.add_node(node_id)
            # G.nodes[node_id]['class'] = node_class
            G.nodes[node_id]['gender'] = node_gender

def readEdges(G, path):
    w = 1
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            G.add_edge(row[0], row[1])
            if len(row) > 2:
                if row[2] in map2:
                    G[row[0]][row[1]]['weight'] = map2[row[2]]
                else:
                    map2[row[2]] = w
                    w += 1
                    G[row[0]][row[1]]['weight'] = map2[row[2]]
            else:
                G[row[0]][row[1]]['weight'] = 1

    print(map2)
