# @title GraphVisualization

import networkx as nx
import matplotlib.pyplot as plt

# TODO
def print_graph(graph: nx.DiGraph, node_colors = None, path_edgelist = None, width_map = None, graph_title: str=None) -> None:
  """
  graph         : wizualizowana sieć, może być graf prosty lub digraf
  node_colors   : sposób pokolorowania wierzchołków
  path_edgelist : lista wyróżnionych krawędzi (w grafie prostym) lub łuków (w digrafie)
  width_map     : sposób pogrubienia krawędzi z listy wyróżnionych krawędzi
  graph_title   : tytuł rysunku
  """

  # Wyświetla podstawowe dane o sieci, chyba że podano listę path_edgelist, tj. krawędzie do wyróżnienia
  # Kolorowanie wierzchołków, domyślnie 'aqua', chyba że podano kolorowanie jako listę node_colors
  color_map = 'aqua'


  # Rysowanie + wyświetlenie tytułu, domyślnie 'Graf zadanej sieci', chyba że podano inny tytuł w graph_title
  fig = plt.figure(figsize=(12, 12))
  plt.title(graph_title or 'Graf zadanej sieci')

  # Rozmieszczenie wierzchołków
  pos = nx.drawing.spring_layout(graph, seed=1)
  # Naniesienie wierzchołków i krawędzi
  nx.draw_networkx(
      graph,
      pos = pos,
      node_color = color_map,
      node_size = 400,
      font_weight = 'bold',
      font_size = 8,
      with_labels = True,
  )

  # Naniesienie etykiet krawędzi - według podanych wag (może ich być dużo, tylko nie będzie to bardzo czytelne)
  edge_labels = {(u,v): '\n'.join(map(lambda x: f'{x[0]}={x[1]}', d.items())) for u,v,d in graph.edges(data=True)}
  nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels)

  # Naniesienie na czerwono i z pogrubieniem wyróżnionych krawędzi, podanych w path_edgelist
  if path_edgelist:

    # Pogrubianie wierzchołków w path_edgeglist, domyślnie 4, chyba że podano jako listę width_map
    width = width_map or 2
    # Rysowanie
    nx.draw_networkx_edges(graph, pos=pos, edgelist=path_edgelist, edge_color='pink', width=width)

  plt.show()

def graphVisualization(sample: any):
  highlight = []
  selected_path = 'P4'
  nested_list = sample['link_to_path'].numpy()
  G = nx.Graph()
  G.add_nodes_from([f'P{i}' for i in range(len(nested_list))], type_='path')
  for data in sample['path_to_link'].numpy():
    if not len(data) == 1:
      for i in range(len(data)-1):
        for j in range(i+1,len(data)):
          if not G.has_edge(f'P{data[i][0]}',f'P{data[j][0]}') or not G.has_edge(f'P{data[j][0]}',f'P{data[i][0]}'):
            G.add_edge(f'P{data[i][0]}',f'P{data[j][0]}')
            if (selected_path == f'P{data[i][0]}' or selected_path == f'P{data[j][0]}'):
              highlight.append((f'P{data[i][0]}',f'P{data[j][0]}'))
  print_graph(G, graph_title = 'Paths dependency graph with higlighted Path #4',path_edgelist = highlight)