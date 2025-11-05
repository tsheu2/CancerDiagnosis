'''
Naive Bayes Classification for Cancer Diagnosis

Objective: To predict the most likely cancer type for a patient based on tumor marker levels
    using the Naive Bayes classifier and Gaussian probability distributions.

    -----------------------------------------------
Formula:
    P(Class | Features) = [ P(Features | Class) * P(Class) ] / P(Features)
Simplified using the Naïve (independence) assumption:
    P(Class | Features) ∝ P(Class) * Π P(x_i | Class)
We compute this for each cancer type and select the class with the highest probability.
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
    for cancer, stats in cancer_stats.items():
        # Start with prior: P(Class)
        prob = 1.0  # Assuming uniform prior for simplicity

        # Multiply by each feature likelihood: Π P(x_i | Class)
        for marker, (mean, var) in stats.items():
            x = patient_data.get(marker, mean)
            prob *= gaussian_probability(x, mean, var)
        
        # Numerator: P(Class) * Π P(x_i | Class)
        probs[cancer] = prob
    
    # Denominator P(Features) = sum over all classes
    total = sum(probs.values())

    # Posterior probability: P(Class | Features)
    for c in probs:
        probs[c] /= total  # Normalize so total = 1

    # Choose class with highest posterior
    prediction = max(probs, key=probs.get)
    return prediction, probs

# Fake Patients w/ tumor marker data
patients = [
    {'name': 'Alice'},
    {'name': 'Bob'},
    {'name': 'Charlie'},
]

