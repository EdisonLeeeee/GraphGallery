import random
import numpy as np
import scipy.sparse as sp
import networkx as nx
from numba import njit
from tqdm import tqdm

class EulerianDiffuser:
    """Class to make diffusions for a given graph.
    """
    def __init__(self, diffusion_cover: int=80, diffusion_number: int=10):
        """
        Parameters:
        -----------
        diffusion_number (int): Number of diffusions
        diffusion_cover (int): Number of nodes in diffusion.
        """
        self.diffusion_number = diffusion_number
        self.diffusion_cover = diffusion_cover

    @staticmethod
    @njit
    def random_diffusion(indices,
                         indptr,
                         diffusion_cover: int = 80,
                         max_steps: int = 1000):
        """
        Generating a diffusion tree from a given source node and linearizing it
        with a directed Eulerian tour.
        """    
        N = len(indptr) - 1
        for n in range(N):
            infected = []
            infected.append(n)
            infected_counter = 1
            edges = []
            steps = 0
            while infected_counter < diffusion_cover:
                randint = random.randint(0, infected_counter - 1)
                end_point = infected[randint]
                neighbors = indices[indptr[end_point]:indptr[end_point + 1]]
                if neighbors.size == 0:
                    break
                sample = np.random.choice(neighbors)         
                if not sample in infected:
                    infected_counter += 1
                    infected.append(sample)
                    edges.append((end_point, sample))
                    edges.append((sample, end_point))
                    if infected_counter == diffusion_cover:
                        break
                elif neighbors.size == 1:
                    # it was stucked in a local neighborhood  
                    break
                    
                steps += 1
                if steps > max_steps:
                    break                    

            yield edges
            
    @staticmethod
    def get_euler(edges, source=None):
        sub_graph = nx.DiGraph()    
        sub_graph.add_edges_from(edges)
        if source is None:
            source = edges[0][0]
        euler = (u for u, v in nx.eulerian_circuit(sub_graph, source))
        yield from euler

    def diffusion(self, graph: sp.csr_matrix):
        # TODO
        # check the graph is connected!
        for _ in tqdm(range(self.diffusion_number), desc='Computing diffusions'):
            edges = self.random_diffusion(graph.indices, graph.indptr, diffusion_cover=self.diffusion_cover)
            for source, edge in enumerate(edges):
                yield list(self.get_euler(edge, source))
