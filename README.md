\# 🪙 Reconnaissance de Pièces Euro — L3 Informatique



> Système de détection et classification automatique de pièces euro par traitement d'image classique (sans deep learning).



\*\*Université Paris Cité — Module Image\*\*  

\*\*Groupe :\*\* Ales Ferhani, Rayane Taouache, Lounes Medjbour, Dania Benhamma



\---



\## 📋 Table des matières



\- \[Présentation](#présentation)

\- \[Pipeline technique](#pipeline-technique)

\- \[Structure du projet](#structure-du-projet)

\- \[Installation](#installation)

\- \[Utilisation](#utilisation)

\- \[Méthodes utilisées](#méthodes-utilisées)

\- \[Résultats](#résultats)

\- \[Limites et perspectives](#limites-et-perspectives)



\---



\## Présentation



Ce projet implémente un système de \*\*reconnaissance automatique de pièces euro\*\* à partir d'images 2D, en utilisant uniquement des méthodes de traitement d'image classiques :



\- Détection des pièces par \*\*transformée de Hough\*\*

\- Classification par \*\*couleur HSV\*\* (bronze / gold / silver)

\- Classification par \*\*k-NN\*\* sur des descripteurs d'histogrammes par anneaux concentriques

\- Estimation du \*\*total en euros\*\* de chaque image



Les 8 dénominations euro sont prises en charge : `1ct`, `2ct`, `5ct`, `10ct`, `20ct`, `50ct`, `1€`, `2€`.



\---



\## Pipeline technique



```

Image

&#x20; │

&#x20; ├─► Resize (≤ 1200px) + GaussianBlur(9×9)

&#x20; │

&#x20; ├─► HoughCircles → détection des cercles

&#x20; │        └─► Déduplication (IoU > 0.7)

&#x20; │

&#x20; ├─► Pour chaque cercle :

&#x20; │        ├─► Crop circulaire masqué

&#x20; │        ├─► Classification couleur HSV (bronze / gold / silver)

&#x20; │        ├─► fitEllipse + correction biais ×1.15 → diamètre en pixels

&#x20; │        └─► Histogrammes HSV + Sobel par anneau → vecteur 240 features

&#x20; │

&#x20; ├─► Scale factor global (pixels → mm)

&#x20; │        └─► Contraint par groupe couleur

&#x20; │

&#x20; ├─► k-NN pondéré (k=5) sur ring\_features

&#x20; │        └─► Prioritaire si confiance > 0.60 (gold) / 0.55 (silver)

&#x20; │

&#x20; ├─► Détection bimétal (centre vs anneau HSV)

&#x20; │        └─► 2€ souvent détecté gold → reclassé silver

&#x20; │

&#x20; └─► Label final + valeur en euros + total

```



\---



\## Structure du projet



```

Projet\_Image/

│

├── core/                        # Modules principaux du pipeline

│   ├── segmentation.py          # HoughCircles + déduplication

│   ├── features.py              # Extraction couleur, ellipse, ring\_features

│   ├── classification.py        # Scale factor + k-NN + bimétal

│   └── utils.py                 # Diamètres officiels, valeurs, draw\_label

│

├── evaluation/                  # Scripts d'évaluation

│   ├── evaluate.py              # Métriques globales (précision, rappel, F1)

│   ├── diagnostic\_labels.py     # Matrice de confusion par classe

│   ├── metrics.py               # IoU, compute\_metrics, normalize\_label

│   └── test\_pipeline.py         # Évaluation sur data/test/

│

├── training/                    # Construction de la base k-NN

│   └── build\_knn\_profiles.py    # Extrait les features depuis les annotations

│

├── demo/                        # Démo visuelle

│   ├── main.py                  # Affichage OpenCV image par image

│   └── demo\_images\_val/         # Images de démonstration

│

├── model/

│   └── knn\_database.npy         # Base k-NN (162 exemples, 240 features)

│

├── data/

│   ├── validation/              # 100 images d'évaluation

│   └── test/                    # 100 images de test

│

├── annotation/                  # Annotations LabelMe (JSON)

│

├── run\_evaluate.py              # Lance l'évaluation sur validation

├── run\_demo.py                  # Lance la démo visuelle

└── README.md

```



\---



\## Installation



\### Prérequis



\- Python 3.9+

\- OpenCV, NumPy



\### Installation des dépendances



```bash

pip install opencv-python numpy

```



\### Cloner le dépôt



```bash

git clone https://github.com/RayaneWebDev/Projet\_Image.git

cd Projet\_Image

```



\---



\## Utilisation



\### Démo visuelle



```bash

python run\_demo.py

```

Lance le pipeline sur toutes les images de `demo\_images\_val/` et affiche le résultat dans une fenêtre OpenCV avec les labels et le total en euros.



\### Évaluation sur la validation



```bash

python run\_evaluate.py

```

Calcule précision, rappel, F1-score et label accuracy sur les 100 images annotées de `data/validation/`.



\### Matrice de confusion



```bash

python evaluation/diagnostic\_labels.py

```

Affiche l'accuracy par classe et les erreurs les plus fréquentes.



\### Évaluation sur le jeu de test



```bash

python evaluation/test\_pipeline.py

```



\### Reconstruire la base k-NN



```bash

python training/build\_knn\_profiles.py

```

À relancer si les features sont modifiées.



\---



\## Méthodes utilisées



\### Segmentation



| Méthode | Rôle |

|---|---|

| `GaussianBlur(9×9)` | Réduction du bruit avant détection |

| `HoughCircles` | Détection des cercles (transformée de Hough) |

| Déduplication par distance | Suppression des détections en double |



\### Extraction de features



| Méthode | Rôle |

|---|---|

| Masque circulaire | Isolation de la pièce sur fond noir |

| Seuillage HSV (`inRange`) | Classification couleur : bronze / gold / silver |

| `fitEllipse` | Correction de la perspective (vue oblique) |

| Correction ×1.15 | Compensation du biais de sous-estimation de HoughCircles |

| Gradient Sobel | Magnitude et direction (invariant à la rotation) |

| Histogrammes par anneau | 3 anneaux × 5 canaux × 16 bins = \*\*240 features\*\* |



\### Classification



| Méthode | Rôle |

|---|---|

| Scale factor global | Conversion pixels → mm par minimisation d'erreur |

| k-NN pondéré (k=5) | Vote par 1/distance sur les ring\_features |

| Détection bimétal | Comparaison centre vs anneau (saturation HSV) |

| Contrainte couleur | Restreint les candidats selon bronze/gold/silver |



\### Évaluation



| Métrique | Description |

|---|---|

| IoU (Intersection over Union) | Chevauchement entre cercle prédit et annoté |

| Précision | TP / (TP + FP) |

| Rappel | TP / (TP + FN) |

| F1-score | Moyenne harmonique précision/rappel |

| Label accuracy | Pièces correctement identifiées / TP |



\---



\## Résultats



\### Métriques globales (100 images de validation)



| Métrique | Score | Seuil | Statut |

|---|---|---|---|

| Précision | \*\*78.6%\*\* | 75% | ✅ |

| Rappel | \*\*75.0%\*\* | 75% | ✅ |

| F1-score | \*\*76.7%\*\* | — | ✅ |

| Label accuracy | \*\*49.4%\*\* | 70% | ❌ |



```

TP=231  FP=63  FN=77

```



\### Accuracy par classe



| Classe | Accuracy |

|---|---|

| 1 Euro | 64% |

| 10 cent | 62% |

| 20 cent | 53% |

| 50 cent | 47% |

| 1 cent | 47% |

| 2 Euro | 48% |

| 5 cent | 29% |

| 2 cent | 25% |



\### Progression au fil des améliorations



| Étape | Label accuracy |

|---|---|

| Baseline (scale factor seul) | 37.7% |

| + k-NN sur anneaux HSV | 42.9% |

| + Correction bimétal gold→silver | 45.0% |

| + Seuils bimétal assouplis | 48.9% |

| + Direction gradient Sobel | \*\*49.4%\*\* |



\---



\## Limites et perspectives



\### Limites fondamentales (méthodes 2D classiques)



\- \*\*10ct / 20ct / 50ct indiscernables\*\* : seulement 2mm d'écart entre chaque, même couleur gold, même gradient, même spectre FFT — confirmé par la littérature (Hossfeld et al., 2006)

\- \*\*Scale factor instable\*\* avec peu de pièces dans l'image

\- \*\*Classification couleur sensible\*\* à l'éclairage (2€ souvent détecté gold)



\### Ce qui a été exploré et écarté



| Méthode | Résultat |

|---|---|

| CLAHE (normalisation éclairage) | Régression |

| Otsu sur contours | Régression |

| Filtre médian | Régression |

| FFT radiale | Aucun gain |

| Transformation polaire + FFT | Aucun gain |

| Moments de Hu | Aucun gain |

| Canny densité de contours | Aucun gain |

| Fréquences dominantes par cercle (Hossfeld 2006) | Aucun gain |



\### Pour dépasser 70% de label accuracy



Dépasser le seuil de 70% nécessiterait :

\- \*\*Deep learning\*\* (YOLO, ResNet) — apprentissage automatique de features discriminantes

\- \*\*Caméra 3D\*\* — exploitation du relief embossé (Hossfeld et al., 2006 : 99.6% avec cette approche)

\- \*\*Dataset plus large\*\* — plus d'exemples annotés pour le k-NN



\---



\## Références



\- Hossfeld et al. — \*Fast 3D-Vision System to Classify Metallic Coins by their Embossed Topography\*, ELCVIA 2006

\- Survey sur la reconnaissance de pièces — \*International Journal of Computer Applications\*, 2012

\- Documentation OpenCV — \[docs.opencv.org](https://docs.opencv.org)

