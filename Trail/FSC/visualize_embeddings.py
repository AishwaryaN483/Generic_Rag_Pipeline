
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_embeddings_tsne_with_query(embeddings, chunks, query_embedding, results):
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Stack embeddings + query
    all_embeddings = np.vstack([embeddings, query_embedding.reshape(1, -1)])

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(all_embeddings)

    chunk_points = reduced[:-1]
    query_point = reduced[-1]

    plt.figure(figsize=(10, 7))

    # Plot all chunks
    plt.scatter(chunk_points[:, 0], chunk_points[:, 1], c="gray", alpha=0.6, label="Chunks")

    # Plot retrieved results in one color
    retrieved_idx = [r["index"] for r in results]
    plt.scatter(chunk_points[retrieved_idx, 0], chunk_points[retrieved_idx, 1],
                c="blue", label="Top Results", s=80)

    # Plot query
    plt.scatter(query_point[0], query_point[1], c="red", marker="*", s=200, label="Query")

    plt.legend()
    plt.title("t-SNE of Embeddings with Query Highlighted")
    plt.show()



def visualize_embeddings_tsne(embeddings, chunks):
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from sklearn.manifold import TSNE

    print("🔄 Running t-SNE dimensionality reduction...")

    X = np.array(embeddings)

    # Reduce to 3D using t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=5, max_iter=2000)
    X_reduced = tsne.fit_transform(X)

    df = pd.DataFrame({
        "x": X_reduced[:, 0],
        "y": X_reduced[:, 1],
        "z": X_reduced[:, 2],
        "chunk": [f"Chunk {i}" for i in range(len(chunks))],
        "preview": [c[:80] + "..." for c in chunks]
    })

    fig = px.scatter_3d(
        df,
        x="x", y="y", z="z",
        color="chunk",
        hover_data={"preview": True, "chunk": True},
        title="t-SNE Visualization of Chunk Embeddings"
    )
    fig.show()

    """
    embeddings: list/array of embeddings (2D)
    chunks: list of chunk texts
    """
    print("🔄 Running t-SNE dimensionality reduction...")

    # Convert to numpy array
    X = np.array(embeddings)

    # Reduce to 3D using t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=5, max_iter=2000)
    X_reduced = tsne.fit_transform(X)

    # Prepare DataFrame for Plotly
    df = pd.DataFrame({
        "x": X_reduced[:, 0],
        "y": X_reduced[:, 1],
        "z": X_reduced[:, 2],
        "chunk": [f"Chunk {i}" for i in range(len(chunks))],
        "preview": [c[:80] + "..." for c in chunks]  # show first 80 chars
    })

    # Interactive 3D scatter plot
    fig = px.scatter_3d(
        df,
        x="x", y="y", z="z",
        color="chunk",
        hover_data={"preview": True, "chunk": True},
        title="t-SNE Visualization of Chunk Embeddings"
    )
    fig.show()
