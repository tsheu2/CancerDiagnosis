What the code does:
- three tumor markers: HE4, AFP, and CA19-9.
- classify a patient into one of several cancer classes/stages:
    - Ovarian: Early, Late (uses HE4)
    - Liver (HCC): Overall, Stage I, Stage II–III, Stage IV (uses AFP)
    - Pancreatic: Overall, Stage I, Stage II–III, Stage IV (uses CA19-9)

For each class, there is one “signal” marker (its medically relevant marker) with a Gaussian (Normal) distribution defined by mean and variance.

For markers not used by that class, the code applies a background penalty by scoring those markers under a “healthy” Gaussian.
This prevents, for example, Pancreatic Stage IV (huge CA19-9) from being misclassified as Ovarian Early just because HE4 is near-normal—the very elevated CA19-9 will look extremely unlikely under the healthy CA19-9 distribution and will penalize the ovarian class.