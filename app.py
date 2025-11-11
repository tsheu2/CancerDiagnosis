'''
Naive Bayes Classification for Cancer Classification Based on Tumor Marker Levels

Objective: Predict the most likely cancer type (or benign) from tumor marker levels
using the Naive Bayes classifier with Gaussian likelihoods.
'''

import math

# ---- Gaussian helper ----
# used to compute log P(x | c)
def logpdf(x: float, mean: float, var: float) -> float:
    '''
    Returns the log of P(tumor marker level (x) | cancer).
    log P(x | mean,var) = -0.5 * [ log(2πvar) + (x - mean)² / var ]
    '''
    return -0.5 * (math.log(2.0 * math.pi * var) + ((x - mean) ** 2) / var)

# ---- Cancer class means and variances ----
# Format: "Class": (Signal Marker, (Mean, Variance))
CLASS_MARKERS = {
    # Ovarian Cancer
    "Ovarian_Early": ("HE4", (151.0, 6_348.0)),
    "Ovarian_Late":  ("HE4", (570.0, 84_840.0)),

    # Liver Cancer
    "Liver_Overall":       ("AFP", (450.0, 2_250_000.0)),
    "Liver_Stage_I":       ("AFP", (100.0, 7_500.0)),
    "Liver_Stage_II_III":  ("AFP", (600.0, 450_000.0)),
    "Liver_Stage_IV":      ("AFP", (6_000.0, 15_000_000.0)),

    # Pancreatic Cancer
    "Pancreatic_Overall":      ("CA19-9", (1_750.0, 3_500_000.0)),
    "Pancreatic_Stage_I":      ("CA19-9", (140.0, 17_500.0)),
    "Pancreatic_Stage_II_III": ("CA19-9", (950.0, 750_000.0)),
    "Pancreatic_Stage_IV":     ("CA19-9", (12_500.0, 35_000_000.0)),
}

ALL_MARKERS = ["HE4", "AFP", "CA19-9"] # used for looping

# ---- Healthy values ----
# Format: "Marker": (Mean, Variance)
# used for non-signal markers in each class
HEALTHY = {
    "HE4":    (60.0, 15.0**2), 
    "AFP":    (5.0,  3.0**2),   
    "CA19-9": (20.0, 10.0**2),  
}

# Assumes uniform prior
# Log P(Class) = -log(number of classes)
LOG_PRIOR = -math.log(len(CLASS_MARKERS))

# Used to compute log-sum-exp for normalization
def _logsumexp(vals):
    # Returns the log of the sum of exponentials of input values
    # log(sum(e^v for v in vals))
    # denominator log P(x) in Bayes formula
    m = max(vals) 
    return m + math.log(sum(math.exp(v - m) for v in vals)) 

def rank_classes(patient):
    '''
    Ranks cancer classes for a given patient based on tumor marker levels.
    Returns a list of (Class, Probability) tuples sorted by probability.
    1. For each class, compute the log-probability score:
       log P(Class) + log P(x_signal | Class) + Σ log P(x_other | Healthy)
    2. Normalize scores to probabilities using log-sum-exp.
    3. Return sorted list of (Class, Probability) tuples.
    '''

    scores = {} # used to store log P(C) + log P(x | C) for each class

    for cls, (signal_marker, (mu, var)) in CLASS_MARKERS.items():
        # loops over each class to compute its score

        x_signal = float(patient[signal_marker]) # extract tumor marker level
        logp = LOG_PRIOR + logpdf(x_signal, mu, var) # log P(Class) + log P(x_marker | Class)

        # Loop over other markers to set as healthy and add their log-probabilities
        for m in ALL_MARKERS:
            if m == signal_marker: # skip signal marker
                continue
            if m in patient and patient[m] is not None:
                h_mu, h_var = HEALTHY[m] # healthy mean/var
                logp += logpdf(float(patient[m]), h_mu, h_var)

        scores[cls] = logp # store log-probability score

    # Normalize to probabilities
    lse = _logsumexp(list(scores.values()))

    # Create ranked list of (Class, Probability) tuples
    # P(Class | x))
    ranked = sorted(((c, math.exp(s - lse)) for c, s in scores.items()), key=lambda kv: kv[1], reverse=True)
    return ranked

# ---- Prediction function ----
def predict_class(patient):
    ranked = rank_classes(patient)
    return ranked[0][0], ranked

# ---- Example patients ----
patients = [
    {'name':'Alice','HE4':180.0,'AFP':6.0,'CA19-9':22.0},      # Ovarian_Early
    {'name':'Charlie','HE4':65.0,'AFP':900.0,'CA19-9':28.0},   # Liver_Stage_II_III
    {'name':'Drew','HE4':70.0,'AFP':8.0,'CA19-9':6000.0},      # Pancreatic_Stage_IV
    {'name':'Evan','HE4':60.0,'AFP':80.0,'CA19-9':30.0},       # Liver_Stage_I
    {'name':'Fiona','HE4':60.0,'AFP':5.0,'CA19-9':12500.0},    # Pancreatic_Stage_IV
    {'name':'Grace','HE4':70.0,'AFP':6000.0,'CA19-9':24.0},    # Liver_Stage_IV
]

# ---- Run classification ----
print("="*60)
print(" Tumor-Marker Naive Bayes Cancer Classification Results ")
print("="*60)
for p in patients:
    pred, ranked = predict_class(p)
    print(f"\nPatient: {p['name']}")
    print(f"  HE4={p['HE4']:.1f}, AFP={p['AFP']:.1f}, CA19-9={p['CA19-9']:.1f}")
    print(f"  -> Predicted Class: ** {pred} **\n")
    print("  Top Predictions:")
    print("  ┌──────────────────────────────┬───────────┐")
    print("  │ Cancer / Stage               │Probability│")
    print("  ├──────────────────────────────┼───────────┤")
    for cls, prob in ranked[:5]:
        print(f"  │ {cls:28s} │ {prob:9.3f} │")
    print("  └──────────────────────────────┴───────────┘")
    print("-"*60)