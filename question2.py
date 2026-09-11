import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# Q2: 16-QAM AND K-MEANS CLUSTERING
# ============================================================

# Store every generated Q2 result inside the Dataset folder
# located next to this script.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")
os.makedirs(OUT, exist_ok=True)

# SNR values used for the 16-QAM experiments.
SNR = [0, 5, 10, 15, 20, 25, 30]


# Use a fresh random generator for reproducible QAM noise.
rng = np.random.RandomState(67)

# Standard square 16-QAM levels normalized so that
# the average symbol energy is approximately one.
levels = np.array(
    [-3, -1, 1, 3]
) / np.sqrt(10)

# Generate the sixteen ideal constellation points.
constellation = np.array(
    [
        (i, q)
        for i in levels
        for q in levels
    ]
)


# ------------------------------------------------------------
# Q2(a): Generate noisy 16-QAM samples
# ------------------------------------------------------------

rows = []

for s in SNR:

    # Complex AWGN standard deviation for Es = 1.
    noise_std = np.sqrt(
        1 / (2 * 10**(s / 10))
    )

    for symbol_id, (i, q) in enumerate(constellation):

        # Generate 200 noisy observations of each constellation point.
        z = (
            i
            + 1j * q
            + noise_std
            * (
                rng.randn(200)
                + 1j * rng.randn(200)
            )
        )

        rows += [
            [
                s,
                symbol_id,
                value.real,
                value.imag,
            ]
            for value in z
        ]


q2 = pd.DataFrame(
    rows,
    columns=[
        "snr_db",
        "symbol_id",
        "xI",
        "xQ",
    ],
)

# Instantaneous amplitude:
#     r = sqrt(xI^2 + xQ^2)
q2["r"] = np.hypot(
    q2.xI,
    q2.xQ,
)

# Instantaneous phase:
#     theta = atan2(xQ, xI)
q2["theta"] = np.arctan2(
    q2.xQ,
    q2.xI,
)

q2.to_csv(
    os.path.join(
        OUT,
        "qam16_awgn_dataset.csv",
    ),
    index=False,
)


# ------------------------------------------------------------
# Helper function: cluster purity
# ------------------------------------------------------------

def purity(y_true, labels):
    """
    Calculate cluster purity using the known transmitted
    16-QAM symbol IDs as ground truth.
    """

    return sum(
        np.bincount(
            y_true[labels == cluster],
            minlength=16,
        ).max()
        for cluster in np.unique(labels)
    ) / len(y_true)


# ------------------------------------------------------------
# Q2(b): Determine the best number of clusters
# ------------------------------------------------------------

# Use only the 25 dB Cartesian data for K analysis.
data_25 = q2[
    q2.snr_db == 25
].reset_index(drop=True)

cartesian_data = data_25[
    ["xI", "xQ"]
].to_numpy()

# Test K from 2 through 20.
k_values = range(2, 21)

inertia_values = []
silhouette_values = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42,
    )

    labels = model.fit_predict(
        cartesian_data
    )

    inertia_values.append(
        model.inertia_
    )

    silhouette_values.append(
        silhouette_score(
            cartesian_data,
            labels,
        )
    )


k_results = pd.DataFrame(
    {
        "K": list(k_values),
        "Inertia": inertia_values,
        "Silhouette coefficient": silhouette_values,
    }
)

k_results.to_csv(
    os.path.join(
        OUT,
        "q2_kmeans_k_analysis.csv",
    ),
    index=False,
)


# ------------------------------------------------------------
# Q2(b): Inertia versus K
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))
plt.plot(k_results.K, k_results.Inertia, marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Inertia vs K")
plt.grid(alpha=0.30)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q2b_inertia_vs_k.png"), dpi=180)
plt.close()


# ------------------------------------------------------------
# Q2(b): Silhouette coefficient versus K
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))
plt.plot(k_results.K, k_results["Silhouette coefficient"], marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette coefficient")
plt.title("Silhouette vs K")
plt.grid(alpha=0.30)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q2b_silhouette_vs_k.png"), dpi=180)
plt.close()


# ------------------------------------------------------------
# Q2(b): Combined inertia and silhouette plot
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].plot(k_results.K, k_results.Inertia, marker="o")
axes[0].set(xlabel="K", ylabel="Inertia", title="Inertia vs K")
axes[1].plot(k_results.K, k_results["Silhouette coefficient"], marker="o")
axes[1].set(xlabel="K", ylabel="Silhouette", title="Silhouette vs K")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "inertia_silhouette_vs_k.png"), dpi=180)
plt.close(fig)


# ------------------------------------------------------------
# Q2(b): K = 16 Cartesian clustering
# ------------------------------------------------------------

model_16 = KMeans(n_clusters=16, init="k-means++", n_init=10, random_state=42)
cluster_labels = model_16.fit_predict(cartesian_data)

plt.figure(figsize=(6, 6))
plt.scatter(cartesian_data[:, 0], cartesian_data[:, 1],
            c=cluster_labels, s=7, alpha=0.35)
plt.scatter(model_16.cluster_centers_[:, 0],
            model_16.cluster_centers_[:, 1],
            marker="X", s=120, edgecolor="black", label="Centroids")
plt.xlabel("$x_I$")
plt.ylabel("$x_Q$")
plt.title("K=16 Cartesian K-means at 25 dB")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "kmeans16_cartesian_scatter.png"), dpi=180)
plt.close()


# ------------------------------------------------------------
# Q2(b): Compare feature representations
# ------------------------------------------------------------

true_symbols = data_25.symbol_id.to_numpy()

# Feature Set 1:
# Cartesian coordinates.
feature_sets = [
    (
        "Cartesian (xI, xQ)",
        data_25[
            ["xI", "xQ"]
        ].to_numpy(),
    ),
    (
        "Polar (r, theta)",
        data_25[
            ["r", "theta"]
        ].to_numpy(),
    ),
]

# Feature Set 3:
# Combined Cartesian + polar features.
# StandardScaler is applied as specified in the assignment.
combined_features = StandardScaler().fit_transform(
    data_25[
        ["r", "xI", "xQ", "theta"]
    ]
)

feature_sets.append(
    (
        "Combined standardized (r, xI, xQ, theta)",
        combined_features,
    )
)

purity_results = []

for feature_name, feature_data in feature_sets:

    model = KMeans(
        n_clusters=16,
        init="k-means++",
        n_init=10,
        random_state=42,
    )

    labels = model.fit_predict(
        feature_data
    )

    purity_results.append(
        [
            feature_name,
            100 * purity(
                true_symbols,
                labels,
            ),
        ]
    )


purity_results = pd.DataFrame(
    purity_results,
    columns=[
        "Feature set",
        "Purity (%)",
    ],
)

purity_results.to_csv(
    os.path.join(
        OUT,
        "q2_feature_set_purity.csv",
    ),
    index=False,
)


# ============================================================
# Q2(c): ADAPTIVE VS FIXED-TEMPLATE K-MEANS
# ============================================================

# Learn the fixed constellation template using the 25 dB data.
fixed_centroids = KMeans(
    n_clusters=16,
    init="k-means++",
    n_init=10,
    random_state=42,
).fit(
    cartesian_data
).cluster_centers_

strategy_results = []


for s in SNR:

    # Extract all received samples at the current SNR.
    current_data = q2[
        q2.snr_db == s
    ]

    samples = current_data[
        ["xI", "xQ"]
    ].to_numpy()

    true_labels = current_data[
        "symbol_id"
    ].to_numpy()

    # --------------------------------------------------------
    # Adaptive K-means
    # --------------------------------------------------------
    # New centroids are learned independently at each SNR.

    adaptive_model = KMeans(
        n_clusters=16,
        init="k-means++",
        n_init=10,
        random_state=42,
    )

    adaptive_labels = adaptive_model.fit_predict(
        samples
    )

    adaptive_purity = 100 * purity(
        true_labels,
        adaptive_labels,
    )

    # --------------------------------------------------------
    # Fixed-template assignment
    # --------------------------------------------------------
    # Reuse the centroids learned from the 25 dB data.
    # Each sample is assigned to its nearest centroid.

    distances = (
        (
            samples[:, None, :]
            - fixed_centroids[None, :, :]
        ) ** 2
    ).sum(axis=2)

    fixed_labels = np.argmin(
        distances,
        axis=1,
    )

    fixed_purity = 100 * purity(
        true_labels,
        fixed_labels,
    )

    strategy_results.append(
        [
            s,
            adaptive_purity,
            fixed_purity,
        ]
    )


strategy_results = pd.DataFrame(
    strategy_results,
    columns=[
        "SNR (dB)",
        "Adaptive K-Means Purity (%)",
        "Fixed Template Purity (%)",
    ],
)

strategy_results.to_csv(
    os.path.join(
        OUT,
        "q2c_strategy_purity.csv",
    ),
    index=False,
)

strategy_results.to_csv(
    os.path.join(
        OUT,
        "adaptive_vs_fixed_purity.csv",
    ),
    index=False,
)


# ------------------------------------------------------------
# Q2(c): Plot adaptive versus fixed-template purity
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(strategy_results.iloc[:, 0], strategy_results.iloc[:, 1],
         marker="o", label="Adaptive K-Means")
plt.plot(strategy_results.iloc[:, 0], strategy_results.iloc[:, 2],
         marker="s", label="Fixed Template")
plt.xlabel("SNR (dB)")
plt.ylabel("Cluster Purity (%)")
plt.title("Adaptive vs Fixed Template")
plt.xticks(SNR)
plt.grid(alpha=0.30)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "adaptive_vs_fixed_purity.png"), dpi=180)
plt.close()


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\nDone. Files saved in:")
print(OUT)

# ============================================================
# Run Question 2
# ============================================================

if __name__ == "__main__":
    pass
