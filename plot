"""
Compare method runtimes as a function of the number of samples,
with a Plotly stacked bar chart.

  x     : number of samples
  y     : runtime (s)
  color : method (stacked)

Runtimes are measured sequentially (one method at a time), so they are not
biased by CPU concurrency. Reuses the reduce_* functions from dr_benchmark.py.
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import normalize

from dr_benchmark import METHODS


def measure_runtimes(X, sample_sizes, random_state=0):
    """
    Time each method for each sample size.

    Args:
        X            : your embeddings, array (n_samples, n_features).
        sample_sizes : sizes to test; each one is a random subsample of X.

    Returns:
        DataFrame indexed by n_samples, columns = methods, values = seconds.
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    sizes = [n for n in sample_sizes if n <= len(X)] or [len(X)]

    rows = {}
    for n in sizes:
        print(f"n_samples = {n}")
        idx = rng.choice(len(X), n, replace=False)
        Xn = X[idx]
        row = {}
        for name, fn in METHODS.items():
            t0 = time.perf_counter()
            fn(Xn)
            row[name] = time.perf_counter() - t0
        rows[n] = row
    df = pd.DataFrame(rows).T
    df.index.name = "n_samples"
    return df.round(4)


def plot_stacked_runtimes(df, title="Runtime per method by number of samples"):
    """Stacked bar: one bar per sample size, one layer per method."""
    x = [str(n) for n in df.index]  # categorical x -> evenly spaced bars
    fig = go.Figure()
    for method in df.columns:
        fig.add_bar(x=x, y=df[method].values, name=method)
    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Number of samples",
        yaxis_title="Runtime (s)",
        legend_title="Method",
        template="plotly_white",
    )
    return fig


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # Replace with your own embeddings:  X = np.load("my_embeddings.npy")
    X, _ = make_blobs(n_samples=5000, n_features=128, centers=8, random_state=0)
    X = normalize(np.asarray(X, dtype=np.float64), norm="l2")

    df = measure_runtimes(X, sample_sizes=[500, 1000, 2000, 4000])
    print("\n=== Runtime (s): n_samples x methods ===")
    print(df.to_string())

    fig = plot_stacked_runtimes(df)
    fig.write_html("dr_runtime_stacked.html")
    print("\nChart -> dr_runtime_stacked.html")
