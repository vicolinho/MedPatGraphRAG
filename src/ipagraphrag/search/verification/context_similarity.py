import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

class ContextSimilarityModule:
    @staticmethod
    def jaccard_similarity(set1, set2):
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union else 1.0

    @staticmethod
    def degree_hist_similarity(g1, g2, bins=10):
        d1 = [d for n, d in g1.degree()]
        d2 = [d for n, d in g2.degree()]
        hist1, _ = np.histogram(d1, bins=bins, range=(0, max(d1 + d2) + 1), density=True)
        hist2, _ = np.histogram(d2, bins=bins, range=(0, max(d1 + d2) + 1), density=True)
        if np.std(hist1) == 0 or np.std(hist2) == 0:
            return 1.0 if np.allclose(hist1, hist2) else 0.0
        return pearsonr(hist1, hist2)[0]

    @staticmethod
    def spectral_similarity(g1, g2, k=10):
        def top_eigs(g, k):
            A = nx.to_numpy_array(g)
            eigs = np.linalg.eigvalsh(A)
            eigs = np.sort(np.abs(eigs))[::-1]
            pad = np.zeros(max(0, k - len(eigs)))
            return np.concatenate([eigs[:k], pad])
        v1 = top_eigs(g1, k)
        v2 = top_eigs(g2, k)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 1.0 if np.allclose(v1, v2) else 0.0
        return 1 - cosine(v1, v2)

    @classmethod
    def composite_similarity(cls, g1, g2):
        node_jac = cls.jaccard_similarity(set(g1.nodes()), set(g2.nodes()))
        edge_jac = cls.jaccard_similarity(set(g1.edges()), set(g2.edges()))
        deg_sim = cls.degree_hist_similarity(g1, g2)
        spec_sim = cls.spectral_similarity(g1, g2)
        # Weighted average (tune as needed)
        weights = [0.25, 0.25, 0.2, 0.3]
        sim = (weights[0]*node_jac + weights[1]*edge_jac +
               weights[2]*deg_sim + weights[3]*spec_sim)
        return sim, dict(node_jac=node_jac, edge_jac=edge_jac, deg=deg_sim, spec=spec_sim)
