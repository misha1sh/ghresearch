import optuna
import networkx as nx
from queue import Queue
import random
import numpy as np
import multiprocessing

from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import json

def calc_graph_numbers(G):
  return {
    'number_of_nodes': G.number_of_nodes(),
    'number_of_edges': G.number_of_edges(),
    'degree_distribution': calc_degree_distribution(G).tolist(),
    'clustering_distribution': calc_clustering_distribution(G).tolist(),
    'bidirectional': calculate_bidirectional_percentage(G),
    'connected': len(list(nx.strongly_connected_components(G))),
  }

def compare_graph_numbers(a, b, weights):
  res = {'total': 0}
  for name, value in weights.items():
    res[name] = np.sum((np.array(a[name]) - np.array(b[name])) ** 2) * value
    res['total'] += res[name]
  return res



# Function to plot degree vs. clustering coefficient
def plot_clustering_coefficient(graph, title, color='b'):
    degrees = np.array([degree for node, degree in graph.degree()])
    clustering_coeffs = np.array(list(nx.clustering(graph).values()))

    # plt.figure()
    plt.scatter(degrees, clustering_coeffs, alpha=0.5, edgecolor=color, label=title)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree')
    plt.ylabel('Clustering Coefficient')
    plt.grid(True)
    # plt.show()

def calc_degree_distribution(G):
  degrees = np.array([degree for node, degree in G.degree()])
  bins = np.logspace(np.log10(1), np.log10(1000), 10)
  hist, bin_edges = np.histogram(degrees, bins=bins, density=False)
  return np.log10(hist + 1.)

def calc_clustering_distribution(G):
  degrees = np.array([degree for node, degree in G.degree()])
  clustering_coeffs = np.array(list(nx.clustering(G).values()))

  degrees_bins = np.logspace(np.log10(1), np.log10(1000), 10)
  clustering_bins = np.logspace(np.log10(0.001), np.log10(1.), 10)
  hist, x_edges, y_edges = np.histogram2d(degrees, clustering_coeffs, bins=[degrees_bins, clustering_bins], density=False)
  return hist


def plot_degree_distribution(graph, use_log=True):
  # degrees = [deg for node, deg in graph.degree()]
  # degree_distribution = np.bincount(degrees) / graph.number_of_nodes()

  # Plot Degree Distribution
  plt.figure(figsize=(10, 6))
  # plt.bar(range(len(degree_distribution)), degree_distribution, width=1)
  degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)
  plt.scatter(*np.unique(degree_sequence, return_counts=True))
  plt.title("Degree Distribution")
  plt.xlabel("Degree")
  plt.ylabel("Count of Nodes")
  if use_log:
    plt.xscale('log')
    plt.yscale('log')

  plt.show()

def bfs_with_edge_limit_and_subgraph(graph, start_node, cnt, max_edges_per_type=200):
    visited = set()
    queue = Queue()
    subgraph_edges = []

    # Initialize the queue with the starting node
    queue.put(start_node)
    visited.add(start_node)
    result = []

    while not queue.empty():
        # Get the current node from the queue
        current_node = queue.get()
        result.append(current_node)
        if len(result) >= cnt: break

        # Process outgoing edges
        outgoing_edges = list(graph.successors(current_node))
        if len(outgoing_edges) > max_edges_per_type:
            outgoing_edges = []

        for neighbor in outgoing_edges:
            subgraph_edges.append((current_node, neighbor))
            if neighbor not in visited:
                visited.add(neighbor)
                queue.put(neighbor)

        # Process incoming edges
        incoming_edges = list(graph.predecessors(current_node))
        if len(incoming_edges) > max_edges_per_type:
            incoming_edges = []

        for neighbor in incoming_edges:
            subgraph_edges.append((neighbor, current_node))
            if neighbor not in visited:
                visited.add(neighbor)
                queue.put(neighbor)

    # Create a subgraph with the visited nodes and their edges
    # subgraph = graph.edge_subgraph(subgraph_edges).copy()
    return graph.subgraph(result)

def convert_to_directed_with_probability(undirected_graph, p1, p2):
    # Create a directed graph
    directed_graph = nx.DiGraph()

    # Iterate over all edges in the undirected graph
    for u, v in undirected_graph.edges():
        if random.random() < p2:
          continue
        directed_graph.add_edge(u, v)
        directed_graph.add_edge(v, u)

        if random.random() < p1:
            # Randomly remove one of the two edges
            if random.choice([True, False]):
                directed_graph.remove_edge(u, v)
            else:
                directed_graph.remove_edge(v, u)

    return directed_graph

def calculate_bidirectional_percentage(G):
    if not G.is_directed():
        raise ValueError("The graph must be directed to calculate bidirectional edges.")

    bidirectional_count = 0
    total_edges = G.number_of_edges()

    for u, v in G.edges():
        if G.has_edge(v, u):
            bidirectional_count += 1

    bidirectional_count //= 2
    bidirectional_percentage = (bidirectional_count / total_edges) if total_edges > 0 else 0
    return bidirectional_percentage


# def count_degree_distr(G):
#   degrees = [deg for node, deg in G.degree()]
#   degree_distribution = np.bincount(degrees) / G.number_of_nodes()
#   blurred = -np.log10(gaussian_filter(degree_distribution, sigma=6.) + 0.01)
#   return blurred


def calc(graph_type, n, m, ppwerlaw_p, p1, p2, dba_m2, dba_p2):
  if graph_type == 'barabasi_albert':
      graph = nx.barabasi_albert_graph(n, m)
  elif graph_type == 'powerlaw_cluster':
      graph = nx.powerlaw_cluster_graph(n, m, ppwerlaw_p)
  elif graph_type == 'dual_barabasi_albert':
      graph = nx.dual_barabasi_albert_graph(n, m, m2=dba_m2, p=dba_p2)

  g = convert_to_directed_with_probability(graph, p1=p1, p2=p2)
  for i in range(min(len(list(g.nodes)), 30)):
    ba_graph = bfs_with_edge_limit_and_subgraph(g, list(g.nodes)[i], cnt=target['number_of_nodes'])
    if ba_graph.number_of_nodes() == target['number_of_nodes']:
      return ba_graph
  return ba_graph
  # return ba_graph.number_of_nodes(), ba_graph.number_of_edges(), len(list(nx.strongly_connected_components(ba_graph)))


# def calc_diff(a1, a2):
#   if a1.shape[0] > a2.shape[0]:
#     return calc_diff(a2, a1)
#   if a1.shape[0] == a2.shape[0]:
#     return a1-a2
#   a1 = np.concatenate((a1, np.zeros((a2.shape[0] - a1.shape[0] ,))))
#   return a1-a2

target = json.loads(open("target.json", 'r').read())
weights = {
    'number_of_nodes': 1.,
    'number_of_edges': 0.1,
    'degree_distribution': 1000.,
    'clustering_distribution': 0.1,
    'bidirectional': 1000000000.,
    'connected': 0.001,
}
def score(trial):
  n = trial.suggest_int('n', target['number_of_nodes'], target['number_of_nodes']*2)
  m = trial.suggest_int('m', 2, 30)
  p1 = trial.suggest_float('p1', 0., 1.)
  p2 = trial.suggest_float('p2', 0., 1.)
  ppwerlaw_p = trial.suggest_float('ppwerlaw_p', 0., 1.)
  # ba_graph = calc(n, m, p1, p2)
  graph_type = trial.suggest_categorical('graph_type', [
    'barabasi_albert', 'powerlaw_cluster',
    'dual_barabasi_albert'
  ])
  dba_m2 = trial.suggest_int('dba_m2', 2, 30)
  dba_p2 =  trial.suggest_float('dba_p2', 0., 1.)
  ba_graph = calc(graph_type, n, m, ppwerlaw_p, p1, p2, dba_m2, dba_p2)
  if ba_graph.number_of_nodes() < target['number_of_nodes'] * 0.9:
    return 1e6 * abs(ba_graph.number_of_nodes() - target['number_of_nodes'])
    # Generate the graph based on the selected graph type

  # degree_dist = count_degree_distr(ba_graph)
  # return ((8296 - ba_graph.number_of_edges()) ** 2 +
  #     (len(list(nx.strongly_connected_components(ba_graph))) - 1349) ** 2 +
  #     (calculate_bidirectional_percentage(ba_graph) - 0.0548) **2 * 100_000 +
  #     np.sum(calc_diff(degree_dist, ideal_degree_dist)**2)*200
  # )
  estimation = calc_graph_numbers(ba_graph)
  return compare_graph_numbers(target, estimation, weights)['total']

def func(study_name):
  study = optuna.load_study(study_name=study_name, storage="mysql://debian-sys-maint:G0sHHzGT6lCby1mu@localhost/studies")
  study.optimize(score, n_trials=1000)

if __name__ == "__main__":
    study = optuna.create_study(storage="mysql://debian-sys-maint:G0sHHzGT6lCby1mu@localhost/studies")
    print(study.study_name)
    # study.optimize(score, n_trials=1000)
    # print(study.best_params)
    processes = []

    for i in range(30):
        process = multiprocessing.Process(target=func, args=(study.study_name,))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()

    study = optuna.load_study(study_name=study_name, storage="mysql://debian-sys-maint:G0sHHzGT6lCby1mu@localhost/studies")
    print(study.best_params)