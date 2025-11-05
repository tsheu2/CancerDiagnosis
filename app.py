'''
Naive Bayes Classification for Cancer Diagnosis

Objective: To predict the most likely cancer type for a patient based on tumor marker levels
    using the Naive Bayes classifier and Gaussian probability distributions.
'''

import numpy as np
import math


# Sample cancer statistics: mean and variance for each tumor marker
cancer_stats = {
    'Prostate Cancer': {},
    'Ovarian Cancer': {},
}


# Gaussian probability density function
def gaussian_probability(x, mean, var):
    return (1 / math.sqrt(2 * math.pi * var)) * math.exp(-(x - mean)**2 / (2 * var))


# Naive Bayes prediction function
def predict(patient_data):
    probs = {}
    for cancer, stats in cancer_stats.items(): # Each cancer type
        prob = 1
        for marker, (mean, var) in stats.items(): # Each tumor marker
            x = patient_data.get(marker, mean) 
            prob *= gaussian_probability(x, mean, var) # Multiply probabilities
        probs[cancer] = prob
    total = sum(probs.values())
    for c in probs: probs[c] /= total  # Normalize probabilities
    prediction = max(probs, key=probs.get)
    return prediction, probs


# Fake Patients w/ tumor marker data
patients = [
    {'name': 'Alice'},
    {'name': 'Bob'},
    {'name': 'Charlie'},
]

