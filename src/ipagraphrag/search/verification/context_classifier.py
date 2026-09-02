import numpy as np
import matplotlib.pyplot as plt

class SufficiencyClassificationModule:
    def __init__(self, ctx_sim_matrix, res_sim_matrix, threshold=0.2):
        self.ctx_sim = ctx_sim_matrix
        self.res_sim = res_sim_matrix
        self.threshold = threshold

    def classify(self):
        n = self.ctx_sim.shape[0]
        verdicts = []
        gap_diffs = []
        for i in range(n):
            for j in range(i+1, n):
                ctx_gap = 1 - self.ctx_sim[i, j]
                res_gap = 1 - self.res_sim[i, j]
                gap_diff = ctx_gap - res_gap
                # Classification logic
                if abs(gap_diff) < self.threshold:
                    label = "SUFFICIENT"
                    explanation = f"Context gap ({ctx_gap:.3f}) ≈ result gap ({res_gap:.3f}): context is being used."
                elif ctx_gap > 0.5 and res_gap < 0.3:
                    label = "INSUFFICIENT"
                    explanation = f"Context gap ({ctx_gap:.3f}) >> result gap ({res_gap:.3f}): context not used."
                elif res_gap > ctx_gap + self.threshold:
                    label = "ANOMALY"
                    explanation = f"Result gap ({res_gap:.3f}) >> context gap ({ctx_gap:.3f}): unexpected pattern."
                else:
                    label = "SUFFICIENT"
                    explanation = f"Context gap ({ctx_gap:.3f}) ≈ result gap ({res_gap:.3f}): context is being used."
                verdicts.append({
                    "pair": (i, j),
                    "ctx_gap": ctx_gap,
                    "res_gap": res_gap,
                    "gap_diff": gap_diff,
                    "label": label,
                    "explanation": explanation
                })
                gap_diffs.append(gap_diff)
        # Aggregate verdict
        n_sufficient = sum(1 for v in verdicts if v["label"] == "SUFFICIENT")
        n_insufficient = sum(1 for v in verdicts if v["label"] == "INSUFFICIENT")
        n_anomaly = sum(1 for v in verdicts if v["label"] == "ANOMALY")
        verdict = "SUFFICIENT" if n_insufficient == 0 else "INSUFFICIENT"
        confidence = 1 - n_insufficient / len(verdicts)
        return verdict, confidence, verdicts, np.mean(gap_diffs), n_sufficient, n_insufficient, n_anomaly

    def plot(self, verdicts, filename="scatter.png"):
        x = [v["ctx_gap"] for v in verdicts]
        y = [v["res_gap"] for v in verdicts]
        colors = {"SUFFICIENT": "green", "INSUFFICIENT": "red", "ANOMALY": "orange"}
        c = [colors[v["label"]] for v in verdicts]
        plt.figure(figsize=(6,6))
        plt.scatter(x, y, c=c, s=80, edgecolor='k')
        plt.plot([0,1], [0,1], 'k--', label="Ideal: ctx_gap = res_gap")
        plt.xlabel("Context Gap (1 - context similarity)")
        plt.ylabel("Result Gap (1 - result similarity)")
        plt.title("Context vs Result Similarity Gaps")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
