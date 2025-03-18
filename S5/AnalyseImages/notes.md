# Analyse d'images

1. Notions: I, IN, TNI, STI, Imagerie
2. Types d’images ?
3. Caractéristiques principales d’une image
4. AI via histogramme
5. AI via primitives (contours, pts d'intérêts, zone d'intérêts)
6. Bruits
7. Apports de l’IA (DL) en AI

## Différents types d'images

1. Binaire : Image noire et blanche = I. 2 niveaux de gris → Matrice dont les éléments prennent les valeurs 0 ou 1 (255) : I ∈ [0, 1].
2. Niveau de gris (I. d'intensité) → Matrice dont les éléments varient entre 0 et 1(255).
3. I. couleur: de base : RGB.

## Caractéristiques principales

1. Résolution spatiale (RS) = taille la plus petite que le détecteur peut distinguer.
    Elle peut être mesurée de plusieurs façons :
    - Nl × Nc
    - Taille d'un pixel (1cm², 1m², …)
    - pouces = 2.54 cm

2. Dynamique : nombre d'intensités
3. Répartition de nombre de pixels en fonction d'intensité (histogramme)
    - Amélioration de la qualité visuelle par des transformations d'histogramme.
    1. Étirement d'histogramme : Min ≤ E ≤ Max → Min - Min ≤ E - Min ≤ Max - Min → 0 ≤ (E - Min) / (Max - Min) × 255 ≤ 255
    2. Égalisation d'histogramme :
