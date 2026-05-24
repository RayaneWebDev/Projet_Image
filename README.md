# Reconnaissance automatique de pièces euro

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-Acad%C3%A9mique-lightgrey?style=flat)

Système de détection et d'identification automatique de pièces euro dans des images photographiées, développé dans le cadre d'un projet de traitement d'image. Le système implémente deux pipelines distincts : une méthode classique par k-NN et une méthode par apprentissage automatique (ExtraTreesClassifier).

**Auteurs :** Ales Ferhani, Rayane Taouache, Lounes Medjbour, Dania Benhamma  
**Université Paris Cité — L3 Informatique — 2025-2026**

---

## Table des matières

1. [Présentation du projet](#présentation-du-projet)
2. [Résultats](#résultats)
3. [Chaîne de traitement](#chaîne-de-traitement)
4. [Métriques d'évaluation](#métriques-dévaluation)
5. [Technologies](#technologies)
6. [Structure du projet](#structure-du-projet)
7. [Installation et lancement](#installation-et-lancement)
8. [Comparaison des approches](#comparaison-des-approches)
9. [Perspectives](#perspectives)
10. [Références](#références)
11. [Auteurs](#auteurs)

---

## Présentation du projet

L'objectif est de détecter et d'identifier automatiquement des pièces euro dans une photographie 2D, sans recours au deep learning. Le système reconnaît les **8 dénominations** en circulation dans la zone euro :

| Groupe chromatique | Dénominations | Diamètre officiel (mm) |
|--------------------|--------------|------------------------|
| Bronze | 1 ct, 2 ct, 5 ct | 16.25 / 18.75 / 21.25 |
| Or | 10 ct, 20 ct, 50 ct | 19.75 / 22.25 / 24.25 |
| Bimétallique | 1 €, 2 € | 23.25 / 25.75 |

Deux approches sont développées et évaluées en parallèle :

- **Méthode classique** : scale factor géométrique (pixels → millimètres) associé à un k-NN pondéré sur des descripteurs par anneaux concentriques.
- **Apprentissage automatique** : mêmes descripteurs, classifiés par un `ExtraTreesClassifier` entraîné indépendamment pour chaque groupe de couleur.

---

## Résultats

Évaluation sur 100 images annotées (jeu de validation), avec un seuil IoU de 0.3 pour valider une détection.

| Méthode | Taux d'identification | Précision | Rappel | F1-score |
|---------|-----------------------|-----------|--------|----------|
| k-NN classique | **73.5 %** | 91.8 % | 83.4 % | 87.4 % |
| ExtraTrees (Random Forest) | **79.4 %** | 91.8 % | 83.4 % | 87.4 % |

```
TP = 257   FP = 23   FN = 51
```

La détection (précision, rappel, F1) est identique pour les deux méthodes car elles partagent la même étape de segmentation. La différence porte exclusivement sur l'identification du label.

---

## Chaîne de traitement

### 1. Numérisation

Chargement de l'image en espace colorimétrique BGR (OpenCV). Si la plus grande dimension dépasse 1 200 px, l'image est redimensionnée par interpolation bilinéaire afin de normaliser le temps de traitement et de limiter les effets de haute résolution sur la détection de cercles.

### 2. Pré-traitements

| Opération | Paramètres | Justification |
|-----------|-----------|---------------|
| Filtre gaussien | Noyau 11×11, σ = 3 | Atténuation du bruit haute fréquence avant la détection de cercles |
| Égalisation d'histogramme | Canal V (HSV) | Normalisation de la luminosité pour robustesse aux variations d'éclairage |
| Filtre moyenneur | Noyau 3×3 | Lissage des crops avant extraction des descripteurs |
| Normalisation z-score | Par canal, par crop | Indépendance à l'intensité lumineuse absolue |

### 3. Segmentation — Transformée de Hough circulaire

```python
cv2.HoughCircles(
    dp      = 1.2,
    minDist = 2 × minRadius,
    param1  = 100,   # seuil Canny haut
    param2  = 55,    # seuil d'accumulation
)
```

La déduplication est réalisée par calcul de distance inter-centres : si deux détections se chevauchent au-delà d'un seuil (IoU > 0.7), seul le cercle de plus grand rayon est conservé.

### 4. Post-traitements par détection

| Opération | Détail |
|-----------|--------|
| Correction de perspective | `cv2.fitEllipse` sur le contour de la pièce, facteur multiplicatif ×1.15 |
| Masque circulaire | Isolation de la pièce sur fond noir |
| Correction de biais | `HoughCircles` sous-estime systématiquement le rayon ; le facteur ×1.15 compense ce biais empirique |

### 5. Extraction de primitives — descripteur `ring_features` (240 dimensions)

Pour chaque pièce, un crop 128×128 normalisé est segmenté en **3 anneaux concentriques**. Sur chaque anneau, 5 histogrammes à 16 bins sont calculés :

| Anneau | Rayon intérieur | Rayon extérieur |
|--------|----------------|----------------|
| Centre | 0 % | 30 % |
| Médian | 30 % | 60 % |
| Bord | 60 % | 100 % |

Les 5 canaux extraits sont : **H**, **S**, **V** (espace HSV), **magnitude du gradient Sobel**, **direction du gradient Sobel**.

Vecteur final : 3 anneaux × 5 canaux × 16 bins = **240 dimensions**.

### 6. Classification

**Méthode classique (k-NN) :**

1. **Scale factor global** : ratio pixels → millimètres estimé en minimisant l'erreur relative sur l'ensemble des pièces de l'image, contraint par groupe de couleur.
2. **k-NN pondéré (k = 5)** : vote pondéré par 1/distance sur une base d'exemples annotés (descripteurs ring_features).
3. **Détection bimétal** : comparaison de la saturation HSV entre le centre (rayon ≤ 20 %) et l'anneau externe (35–44 %). Les pièces classifiées or avec un centre nettement plus saturé sont reclassifiées en bimétalliques (1 € ou 2 €).

**Méthode ExtraTrees :**

Même pipeline (scale factor, bimétal), mais le k-NN est remplacé par un `ExtraTreesClassifier` entraîné séparément pour chaque groupe chromatique. La décision entre scale factor et ExtraTrees est régie par un seuil de confiance :

| Groupe | Seuil de confiance ExtraTrees |
|--------|-------------------------------|
| Or | > 0.55 |
| Bronze | > 0.60 |
| Bimétallique | > 0.55 (séparation finale 1 €/2 € par détection bimétal) |

---

## Métriques d'évaluation

| Métrique | Définition | Rôle dans l'évaluation |
|----------|-----------|------------------------|
| IoU (Intersection over Union) | Rapport entre l'intersection et l'union de deux cercles | Valide qu'une détection correspond à une pièce annotée (seuil : 0.3) |
| Précision | TP / (TP + FP) | Proportion de détections effectivement correctes |
| Rappel | TP / (TP + FN) | Proportion de pièces annotées effectivement détectées |
| F1-score | 2 × (Précision × Rappel) / (Précision + Rappel) | Compromis détection / couverture |
| Taux d'identification | TP correctement labellisés / TP total | Qualité de la classification parmi les vraies détections |

---

## Technologies

| Bibliothèque | Version | Usage |
|-------------|---------|-------|
| OpenCV | 4.x | Segmentation, extraction de features, traitement d'image |
| NumPy | 1.x | Calculs vectoriels, manipulation des descripteurs |
| scikit-learn | 1.x | ExtraTreesClassifier, métriques |
| PyTorch | 2.x | CNN et MobileNet (comparaison expérimentale, non déployé) |
| customtkinter | — | Interface graphique |
| LabelMe | — | Annotation manuelle des images d'entraînement |

---

## Structure du projet

```
Projet_Image/
│
├── core/                         # Modules du pipeline principal
│   ├── segmentation.py           # HoughCircles, déduplication, masque circulaire
│   ├── features.py               # Classification couleur HSV, ellipse, ring_features
│   ├── classification.py         # Scale factor + k-NN pondéré + détection bimétal
│   ├── classification_ml.py      # Variante ExtraTrees par groupe chromatique
│   └── utils.py                  # Diamètres officiels, valeurs faciales, draw_label
│
├── evaluation/                   # Outils de mesure des performances
│   ├── evaluate.py               # Boucle d'évaluation et calcul des métriques globales
│   ├── metrics.py                # IoU, compute_metrics, normalisation des labels
│   ├── diagnostic_labels.py      # Matrice de confusion par classe
│   └── test_pipeline.py          # Évaluation sur data/test/
│
├── training/                     # Construction des bases d'exemples
│   └── build_knn_profiles.py     # Génère knn_database.npy et knn_database_rf.npy
│
├── demo/                         # Démonstrations visuelles (affichage OpenCV)
│   ├── main.py                   # Pipeline classique, image par image
│   └── main_rf.py                # Pipeline ExtraTrees, image par image
│
├── model/                        # Bases d'exemples sérialisées
│   ├── knn_database.npy          # Base k-NN classique (240 features par exemple)
│   └── knn_database_rf.npy       # Base ExtraTrees (240 features, float32)
│
├── data/
│   ├── validation/               # 100 images annotées pour l'évaluation
│   └── test/                     # Images de test complémentaires
│
├── annotation/                   # Fichiers JSON LabelMe, un par image annotée
│
├── app.py                        # Interface graphique customtkinter
├── run_evaluate.py               # Lance l'évaluation de la méthode k-NN
├── run_evaluate_rf.py            # Lance l'évaluation de la méthode ExtraTrees
├── run_demo.py                   # Démonstration visuelle du pipeline classique
└── run_demo_rf.py                # Démonstration visuelle du pipeline ExtraTrees
```

---

## Installation et lancement

### Prérequis

- Python 3.9 ou supérieur
- Compatible Windows, macOS et Linux

### Cloner le dépôt et installer les dépendances

```bash
git clone https://github.com/RayaneWebDev/Projet_Image.git
cd Projet_Image

pip install opencv-python numpy scikit-learn torch torchvision customtkinter pillow
```

### Construction de la base d'exemples

```bash
# Génère model/knn_database.npy et model/knn_database_rf.npy
python training/build_knn_profiles.py
```

> **Attention :** cette commande écrase les fichiers de modèle existants. Ne l'exécuter qu'après modification de la chaîne d'extraction de features, puis vérifier que les scores d'évaluation restent stables.

### Évaluation des performances

```bash
# Méthode classique — k-NN
python run_evaluate.py

# Méthode machine learning — ExtraTrees
python run_evaluate_rf.py
```

### Interface graphique

```bash
python app.py
```

### Démonstrations visuelles

```bash
# Pipeline classique
python run_demo.py

# Pipeline ExtraTrees
python run_demo_rf.py
```

---

## Comparaison des approches

| Critère | k-NN classique | ExtraTrees |
|---------|---------------|------------|
| Taux d'identification | 73.5 % | **79.4 %** |
| Entraînement requis | Non (base d'exemples) | Oui (ajustement des arbres) |
| Exemples d'entraînement | 325 | 325 |
| Sensibilité aux hyperparamètres | Faible | Modérée |
| Explicabilité | Élevée (vote par voisinage) | Modérée (importance des features) |
| Temps d'inférence | Rapide | Rapide |

**Analyse :** avec 325 exemples annotés (soit ~40 par classe), l'ExtraTrees améliore le taux d'identification de 5.9 points par rapport au k-NN. Le k-NN reste compétitif car il ne nécessite pas de généralisation au sens statistique : il mémorise les exemples et vote directement par proximité géométrique dans l'espace des features.

À ce volume de données, les deux approches atteignent la même limite fondamentale : la variabilité intra-classe due à l'éclairage, à l'angle de prise de vue et aux motifs nationaux dépasse la capacité discriminante des descripteurs ring_features.

La principale source d'erreur résiduelle est la confusion au sein du groupe or (10 ct / 20 ct / 50 ct) : ces trois pièces partagent la même couleur et des diamètres séparés de seulement 2 mm, ce qui les rend quasiment indiscernables en vision 2D sans information de profondeur.

---

## Perspectives

- **Augmentation du corpus** : un ensemble de 500 exemples par classe permettrait aux méthodes d'apprentissage automatique de mieux généraliser aux conditions de prise de vue non représentées.
- **Transfer learning** : un modèle léger pré-entraîné (MobileNetV3, EfficientNet-Lite) fine-tuné sur les 8 classes pourrait capturer les textures à haute fréquence que les histogrammes ne discriminent pas.
- **Vision 3D** : l'exploitation du relief embossé via une caméra de profondeur atteindrait théoriquement 99.6 % d'identification (Hossfeld et al., 2006).
- **Déploiement embarqué** : intégration dans une application mobile avec inférence locale (ONNX Runtime, TensorFlow Lite).

---

## Références

- M. Hossfeld et al., *Fast 3D-Vision System to Classify Metallic Coins by their Embossed Topography*, ELCVIA, 2006
- K. Khashman et al., *Automatic coin recognition and classification*, International Journal of Computer Applications, 2012
- Documentation OpenCV — [docs.opencv.org](https://docs.opencv.org)

---

**Université Paris Cité — L3 Informatique — Module Traitement d'image — 2025-2026**
