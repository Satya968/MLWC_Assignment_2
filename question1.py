import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# GLOBAL SETTINGS
# ============================================================

# Store every generated result inside a Dataset folder located
# next to this Python script.
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")
os.makedirs(OUT, exist_ok=True)

# SNR values used throughout both questions.
SNR = [0, 5, 10, 15, 20, 25, 30]

# Number of channel taps.
L = 6

# Random generator used for the 16-QAM AWGN data.
rng = np.random.RandomState(67)


# ============================================================
# Q1: LOS/NLOS CLASSIFICATION
# ============================================================

# The LOS/NLOS dataset is already generated.
# It must be placed in the same directory as this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "los_nlos_dataset.csv")


def q1_dataset():
    """Load the existing LOS/NLOS dataset."""

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Could not find los_nlos_dataset.csv in:\n{SCRIPT_DIR}\n\n"
            "Put the already-generated los_nlos_dataset.csv in the same "
            "folder as assignment2_solution.py."
        )

    return pd.read_csv(INPUT_CSV)


def q1_features(df):
    """
    Extract the five channel features required for Q1.

    Features:
        1. Kurtosis
        2. Skewness
        3. Rising time
        4. RMS delay spread
        5. Rician K-factor
    """

    # Names of the real, imaginary, and delay columns for
    # the six channel taps.
    real_cols = [f"h_real_{i}" for i in range(L)]
    imag_cols = [f"h_imag_{i}" for i in range(L)]
    tau_cols = [f"tau_{i}" for i in range(L)]

    # Convert the required dataframe columns to NumPy arrays
    # for efficient numerical processing.
    real_array = df[real_cols].to_numpy(dtype=float)
    imag_array = df[imag_cols].to_numpy(dtype=float)
    tau_array = df[tau_cols].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Received power of each channel tap
    # --------------------------------------------------------

    P = real_array**2 + imag_array**2

    # --------------------------------------------------------
    # 1. Kurtosis and 2. Skewness
    # --------------------------------------------------------
    # Both features are calculated from the received POWER
    # distribution across the six channel taps.

    kurtosis = stats.kurtosis(
        P,
        axis=1,
        fisher=True,
        bias=True,
    )

    skewness = stats.skew(
        P,
        axis=1,
        bias=True,
    )

    # --------------------------------------------------------
    # 3. Rising time
    # --------------------------------------------------------
    # Rising time is defined as the time required for the
    # cumulative received energy to increase from 10% to 90%.
    #
    # First, sort the channel taps according to delay.

    order = np.argsort(tau_array, axis=1)

    tau_sorted = np.take_along_axis(
        tau_array,
        order,
        axis=1,
    )

    P_sorted = np.take_along_axis(
        P,
        order,
        axis=1,
    )

    # Calculate cumulative and normalized received energy.
    cum_energy = np.cumsum(P_sorted, axis=1)

    cum_norm = cum_energy / cum_energy[:, -1:]


    def interp_threshold(thresh):
        """Interpolate the delay corresponding to a given energy threshold."""

        result = np.empty(len(df))

        for i in range(len(df)):
            result[i] = np.interp(
                thresh,
                cum_norm[i],
                tau_sorted[i],
            )

        return result


    # Find the 10% and 90% cumulative-energy delays.
    t10 = interp_threshold(0.10)
    t90 = interp_threshold(0.90)

    rising_time = t90 - t10

    # --------------------------------------------------------
    # 4. RMS delay spread
    # --------------------------------------------------------

    total_energy = P.sum(axis=1)

    # Power-weighted mean delay.
    mean_tau = (
        np.sum(P * tau_array, axis=1)
        / total_energy
    )

    # Power-weighted RMS delay spread.
    rms_delay_spread = np.sqrt(
        np.sum(
            P * (tau_array - mean_tau[:, None])**2,
            axis=1,
        )
        / total_energy
    )

    # --------------------------------------------------------
    # 5. Rician K-factor
    # --------------------------------------------------------
    # K = dominant-tap power / scattered power.
    #
    # The dominant component is the strongest channel tap.
    # Scattered power is the remaining received power.

    P_dominant = P.max(axis=1)

    P_scattered = (
        P.sum(axis=1)
        - P_dominant
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        rician_k = np.where(
            P_scattered > 0,
            P_dominant / P_scattered,
            np.inf,
        )

    # Return all five features as a single NumPy array.
    return np.column_stack(
        [
            kurtosis,
            skewness,
            rising_time,
            rms_delay_spread,
            rician_k,
        ]
    )


def svm(x, y, train_idx, test_idx):
    """
    Train an RBF-kernel SVM after standardizing the input features.

    The scaler is fitted only on the training data to avoid
    information leakage.
    """

    scaler = StandardScaler().fit(x[train_idx])

    model = SVC(
        C=1,
        kernel="rbf",
    ).fit(
        scaler.transform(x[train_idx]),
        y[train_idx],
    )

    predictions = model.predict(
        scaler.transform(x[test_idx])
    )

    accuracy = 100 * np.mean(
        predictions == y[test_idx]
    )

    return accuracy, scaler, model


# ------------------------------------------------------------
# Q1(a): Load dataset and calculate the five features
# ------------------------------------------------------------

print("Q1: loading existing los_nlos_dataset.csv...")

d = q1_dataset()

# Save a copy of the input dataset inside the output folder.
d.to_csv(
    os.path.join(OUT, "los_nlos_dataset.csv"),
    index=False,
)

# Calculate the five requested channel features.
X = q1_features(d)

names = [
    "Kurtosis",
    "Skewness",
    "Rising time (ns)",
    "RMS delay spread (ns)",
    "Rician K-factor",
]

# Add the extracted features to the dataframe.
for i, name in enumerate(names):
    d[name] = X[:, i]

# Save the complete augmented dataset.
d.to_csv(
    os.path.join(OUT, "los_nlos_dataset_with_features.csv"),
    index=False,
)

# Save a compact file containing only SNR, label, and features.
d[
    ["snr_db", "label"] + names
].to_csv(
    os.path.join(OUT, "q1_features.csv"),
    index=False,
)

y = d.label.to_numpy()
snr = d.snr_db.to_numpy()


# ------------------------------------------------------------
# Q1(b): Train six SVM classifiers at every SNR
# ------------------------------------------------------------

results = []
splits = {}

for s in SNR:

    # Select samples belonging to the current SNR.
    indices = np.where(snr == s)[0]

    # Use the same stratified 80/20 split for all six classifiers
    # so that their accuracies can be compared fairly.
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=y[indices],
        random_state=42,
    )

    splits[s] = (train_idx, test_idx)

    # Train one SVM for each individual feature.
    for j, name in enumerate(names):

        accuracy, _, _ = svm(
            X[:, j:j + 1],
            y,
            train_idx,
            test_idx,
        )

        results.append(
            [s, name, accuracy]
        )

    # Train the sixth SVM using all five features.
    accuracy, _, _ = svm(
        X,
        y,
        train_idx,
        test_idx,
    )

    results.append(
        [s, "All five features", accuracy]
    )


# Convert results to a dataframe.
r1 = pd.DataFrame(
    results,
    columns=[
        "SNR (dB)",
        "Classifier",
        "Accuracy (%)",
    ],
)

# Save the Q1(b) results.
r1.to_csv(
    os.path.join(OUT, "q1_svm_accuracy.csv"),
    index=False,
)

r1.to_csv(
    os.path.join(OUT, "svm_feature_analysis_results.csv"),
    index=False,
)


# ------------------------------------------------------------
# Q1(b): Plot accuracy versus SNR
# ------------------------------------------------------------

pivot = r1.pivot(index="SNR (dB)", columns="Classifier", values="Accuracy (%)")
plot_info = [
    ("Kurtosis", "SVM-1: Kurtosis", "o"),
    ("Skewness", "SVM-2: Skewness", "s"),
    ("Rising time (ns)", "SVM-3: Rise time", "^"),
    ("RMS delay spread (ns)", "SVM-4: RMS delay spread", "D"),
    ("Rician K-factor", "SVM-5: Rician K-factor", "v"),
    ("All five features", "SVM-6: All features", "*"),
]

plt.figure(figsize=(13, 8))
for column, label, marker in plot_info:
    plt.plot(pivot.index, pivot[column], marker=marker, linewidth=2.5,
             markersize=8, label=label)

plt.xlabel("SNR (dB)", fontsize=14)
plt.ylabel("Classification Accuracy (%)", fontsize=14)
plt.title("LOS/NLOS Classification Accuracy vs. SNR "
          "(per feature vs. combined)", fontsize=18)
plt.xticks(SNR, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(59.5, 101.5)
plt.grid(True, alpha=0.30)
plt.legend(loc="lower right", fontsize=12, framealpha=0.90)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "accuracy_vs_snr.png"), dpi=180)
plt.close()


# ------------------------------------------------------------
# Q1(c): Compare training strategies
# ------------------------------------------------------------

# Train the reference classifier using only the 25 dB samples.
train_25, test_25 = splits[25]

accuracy_25, scaler_25, model_25 = svm(
    X,
    y,
    train_25,
    test_25,
)

comparison = []

for s in SNR:

    train_idx, test_idx = splits[s]

    # Strategy 1:
    # Train at the same SNR as the test data.
    matched_accuracy, _, _ = svm(
        X,
        y,
        train_idx,
        test_idx,
    )

    # Strategy 2:
    # Train only at 25 dB and test at the current SNR.
    accuracy_25_model = 100 * np.mean(
        model_25.predict(
            scaler_25.transform(X[test_idx])
        )
        == y[test_idx]
    )

    comparison.append(
        [
            s,
            matched_accuracy,
            accuracy_25_model,
        ]
    )


comparison = pd.DataFrame(
    comparison,
    columns=[
        "SNR (dB)",
        "Train = Test SNR Accuracy (%)",
        "Train at 25 dB Accuracy (%)",
    ],
)

comparison.to_csv(
    os.path.join(OUT, "q1c_training_strategy.csv"),
    index=False,
)

comparison.to_csv(
    os.path.join(
        OUT,
        "part_c_train25_vs_matched_results.csv",
    ),
    index=False,
)


# ------------------------------------------------------------
# Q1(c): Plot training-strategy comparison
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(comparison.iloc[:, 0], comparison.iloc[:, 1],
         marker="o", label="Train = Test SNR")
plt.plot(comparison.iloc[:, 0], comparison.iloc[:, 2],
         marker="s", label="Train at 25 dB")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Q1(c): Training Strategy")
plt.xticks(SNR)
plt.grid(alpha=0.30)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "part_c_train25_vs_matched.png"), dpi=180)
plt.close()

# ============================================================
# Run Question 1
# ============================================================

if __name__ == "__main__":
    pass
