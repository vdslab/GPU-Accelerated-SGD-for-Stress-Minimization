# Graph Drawing by Stochastic Gradient Descent
# https://arxiv.org/abs/1710.04626

from collections import deque
import random
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def calc_adj_matrix(H):
  n = H.number_of_nodes()
  adj = [[] for _ in range(n)]
  for edge in H.edges():
    adj[edge[0]].append(edge[1])
    adj[edge[1]].append(edge[0])
  return adj

def calc_dist_matrix(H):
  adj = calc_adj_matrix(H)
  n = len(adj)
  dist_matrix = {}
  for i in range(n):
    deq = deque()
    seen = [False] * n
    d = {}
    deq.append(i)
    seen[i] = True
    d[i] = 0
    while deq:
      v = deq.popleft()
      for u in adj[v]:
        if seen[u]:
          continue
        deq.append(u)
        seen[u] = True
        d[u] = d[v] + 1
    dist_matrix[i] = d
  return dist_matrix

def calc_edge_info(H, dist):
  nodes = list(H.nodes())
  pairs = []
  dmin = math.inf
  dmax = 0.0
  for u in dist:
    for v in dist[u]:
      if u>=v:
        continue
      dij = float(dist[u][v])
      if dij<=0:
        continue
      wij = 1.0/(dij*dij)
      pairs.append((u,v,dij,wij))
      dmin = min(dmin,dij)
      dmax = max(dmax,dij)
      
  wmin = 1.0/(dmax*dmax)
  wmax = 1.0/(dmin*dmin)
  return pairs,wmin,wmax,nodes
  
def calc_learning_rate(tmax,wmin,wmax,eps=0.1):
  # NOTE: 学習率の境界について
  # 最弱の制約でも μ = 1になるように(他は1を超える1にキャップする)
  # 最強の制約でも μ = ε に収まるように設定(他はさらに小さい)
  eta_max = 1.0/wmin
  eta_min = eps/wmax
  lamb = math.log(eta_max/eta_min)/(tmax-1)
  etas = [eta_max*math.exp(-lamb*t) for t in range(tmax)]
  return etas

def calc_dist(diff):
  return math.sqrt(diff[0]**2 + diff[1]**2)

def sgd(H,dim=2,iterations=15,epsilon=0.1,seed=0,center=True):
  rng = np.random.RandomState(seed)
  
  # NOTE: 前処理
  dist = calc_dist_matrix(H)
  pairs,wmin,wmax,nodes = calc_edge_info(H,dist)
  etas = calc_learning_rate(iterations,wmin,wmax,eps=epsilon)
  
  # NOTE: 初期配置を計算と保存
  n = len(nodes)
  X = rng.rand(n,dim)
  if center:
    X -= X.mean(axis=0,keepdims=True)
  pos_init = {nodes[i]:X[i].copy() for i in range(n)}
  
  # NOTE: SGDを実行
  tiny = 1e-12
  for eta in etas:
    random.shuffle(pairs)
    for i,j,dij,wij in pairs:
      diff = X[j]-X[i]
      norm = calc_dist(diff)
      if norm<tiny:
        diff = rng.normal(scale=1e-6,size=dim)
        norm = calc_dist(diff)
        
      # NOTE: i から 勾配方向に ずれ*学習率*(1/2) ずつ移動
      r = ((norm-dij)/2.0)*(diff/norm)
      mu = min(wij*eta,1.0)
      X[i] += mu*r
      X[j] -= mu*r
      
  if center:
    X -= X.mean(axis=0,keepdims=True)
          
  pos_final = {nodes[i]: X[i].copy() for i in range(n)}
  return pos_init, pos_final

def calc_stress(H, pos):
  dist = calc_dist_matrix(H)
  nodes = list(H.nodes())
  X = np.array([pos[u] for u in nodes], dtype=float)
  val = 0.0
  for u in dist:
    for v in dist[u]:
      if u>=v:
        continue
      dij = float(dist[u][v])
      wij = 1.0/(dij*dij)
      # NOTE: 1/2にすることで、勾配の計算が楽になる
      val += (1/2.0) * wij * (calc_dist(X[u]-X[v]) - dij)**2
  return val
  
if __name__ == "__main__":
  # NOTE: グラフ設定
  n,p,seed = 500,0.005,34
  base = nx.random_graphs.fast_gnp_random_graph(n,p,seed)
  
  # NOTE: リラベル変換(前処理)
  # リラベル（nxはノード番号が連番でない可能性があるため、0からの連番に変換する）
  H = nx.convert_node_labels_to_integers(base,first_label=0,label_attribute="orig_label")
  # リラベルの復元用
  # orig = nx.get_node_attributes(H, "orig_label")
  
  pos0, pos1 = sgd(H,iterations=15,epsilon=0.1,seed=0)

  s0 = calc_stress(H,pos0)
  s1 = calc_stress(H,pos1)
  
  print(f"stress (init) = {s0:.3f}")
  print(f"stress (after) = {s1:.3f}")

  fig, axes = plt.subplots(1, 2, figsize=(30,10))
  axes[0].set_title(f"Initial (stress={s0:.2f})")
  nx.draw(H, pos0, ax=axes[0], node_size=30, width=0.8, with_labels=False)
  axes[0].axis("equal")

  axes[1].set_title(f"After SGD (stress={s1:.2f})")
  nx.draw(H, pos1, ax=axes[1], node_size=30, width=0.8, with_labels=False)
  axes[1].axis("equal")

  plt.tight_layout()
  plt.show()