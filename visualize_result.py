#!/usr/bin/env python3
"""
Visualize vram-lock results and calculate stress
"""

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def read_result_file(filepath):
    """Read vram-lock result file"""
    edges = []
    positions = []
    node_count = 0
    edge_count = 0
    
    with open(filepath, 'r') as f:
        mode = None
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Parse metadata
            if line.startswith('# Node count:'):
                node_count = int(line.split(':')[1].strip())
            elif line.startswith('# Edge count:'):
                edge_count = int(line.split(':')[1].strip())
            elif line.startswith('# Edges'):
                mode = 'edges'
                continue
            elif line.startswith('# Positions'):
                mode = 'positions'
                continue
            elif line.startswith('#'):
                continue
            
            # Parse data
            if mode == 'edges':
                parts = line.split()
                if len(parts) == 2:
                    edges.append((int(parts[0]), int(parts[1])))
            elif mode == 'positions':
                parts = line.split()
                if len(parts) == 2:
                    positions.append([float(parts[0]), float(parts[1])])
    
    return node_count, edges, np.array(positions)

def calc_stress(G, pos):
    """Calculate stress (same as sgd_stress_nongpu.py)"""
    import scipy.sparse.csgraph as csg
    n = len(pos)
    A = nx.adjacency_matrix(G).todense()
    D = csg.shortest_path(A, method='FW', directed=False)
    
    stress = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dij = D[i,j]
            if dij == 0 or np.isinf(dij):
                continue
            xi, yi = pos[i]
            xj, yj = pos[j]
            euc = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            stress += (euc - dij)**2
    return stress

def visualize(filepath, output_image=None):
    """Visualize result and calculate stress"""
    # Read result file
    node_count, edges, positions = read_result_file(filepath)
    
    print(f"File: {filepath}")
    print(f"Nodes: {node_count}, Edges: {len(edges)}")
    
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(node_count))
    G.add_edges_from(edges)
    
    # Create position dict
    pos_dict = {i: positions[i] for i in range(len(positions))}
    
    # Calculate stress
    stress = calc_stress(G, positions)
    print(f"Stress: {stress:.3f}")
    
    # Visualize
    plt.figure(figsize=(12, 10))
    plt.title(f"vram-lock Result (stress={stress:.2f})")
    nx.draw(G, pos_dict, node_size=5, width=0.5, with_labels=False)
    plt.axis("equal")
    
    if output_image:
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        print(f"Image saved to {output_image}")
    else:
        plt.show()
    
    return stress

if __name__ == "__main__":
    # Interactive mode: prompt for input
    print("=== Result Visualizer ===")
    print("Enter result file basename (e.g., vram-lock-20251208_063056 or python-sgd-20251208_063459)")
    print("Or enter full path (e.g., output/vram-lock-20251208_063056.txt)")
    print()
    
    basename = input("ファイル名を入力してください: ").strip()
    
    if not basename:
        print("Error: No filename provided")
        sys.exit(1)
    
    # Auto-complete path: add output/ prefix and .txt suffix if needed
    if not basename.startswith('output/'):
        result_file = f"output/{basename}"
    else:
        result_file = basename
    
    if not result_file.endswith('.txt'):
        result_file = f"{result_file}.txt"
    
    # Check if file exists
    if not Path(result_file).exists():
        print(f"Error: File not found: {result_file}")
        sys.exit(1)
    
    # Auto-generate output image name
    base_without_ext = Path(result_file).stem
    output_image = f"output/{base_without_ext}.png"
    
    print()
    visualize(result_file, output_image)


