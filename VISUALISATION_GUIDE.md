# Guide Complet des Visualisations - Deep3D-OrganSeg

Ce document décrit toutes les visualisations disponibles dans le projet, conformes aux standards de l'imagerie médicale R&D.

## 📊 1. Graphiques Analytiques (Training Analysis)

### Script dédié : `scripts/plot_training_curves.py`

Génère les graphiques standards pour l'analyse de training :

**Graphiques générés :**
- ✅ **Training Loss / Validation Loss** (`loss_curves.png`)
- ✅ **Training Dice / Validation Dice** (`dice_curves.png`)
- ✅ **Training vs Validation (combiné)** (`training_curves.png`)
- ✅ **Hausdorff Distance (optionnel)** (`hausdorff_curve.png`)

**Utilisation :**
```bash
# Générer tous les graphiques
python scripts/plot_training_curves.py --log_path logs/training_history_20240101_120000.json --output_dir outputs/

# Avec Hausdorff
python scripts/plot_training_curves.py --log_path logs/training_history_20240101_120000.json --output_dir outputs/ --include_hausdorff
```

**Notebook :** `notebooks/03_training_analysis.ipynb`
- Analyse interactive des logs de training
- Visualisation des courbes Dice, Loss, IoU
- Comparaison training vs validation
- Statistiques récapitulatives

---

## 🖼️ 2. Visualisation 2D Médicale (3 Plans Standards)

### Notebook : `notebooks/04_inference_visualization.ipynb`

**Fonctionnalités :**
- ✅ **3 vues standards** : Axial, Coronal, Sagittal
- ✅ **Overlay segmentation** : Volume + masque superposé
- ✅ **Navigation interactive** : Exploration de slices spécifiques
- ✅ **Comparaison avec ground truth** : Si disponible

**Fonction principale :**
```python
plot_three_views(volume, segmentation, slice_indices=None)
```

**Affiche :**
- Volume en niveaux de gris
- Segmentation en rouge (overlay transparent)
- 3 plans simultanément pour validation clinique

**Prototype Streamlit :** `app.py`
- Interface web pour upload de volumes
- Affichage des 3 plans avec slider interactif
- Segmentation en temps réel

---

## 🎮 3. Visualisation 3D Interactive (Napari)

### Standard de l'industrie : Napari

**Notebooks avec visualisation Napari :**

1. **`notebooks/01_data_exploration.ipynb`**
   - Exploration du volume brut
   - Volume + masque (si disponible)

2. **`notebooks/02_preprocessing_check.ipynb`**
   - Comparaison brut vs prétraité
   - Visualisation des effets du preprocessing

3. **`notebooks/04_inference_visualization.ipynb`**
   - Volume + prédiction
   - Comparaison prédiction vs ground truth

**Fonctionnalités Napari :**
- ✅ Scroll dans les 3 axes (molette souris)
- ✅ Zoom et pan (clic + drag)
- ✅ Ajustement contraste/brightness
- ✅ Toggle layers on/off
- ✅ Ajustement opacité par couche
- ✅ Spacing correct pour aspect ratio anatomique

**Code standard :**
```python
import napari

viewer = napari.Viewer(title="Volume + Segmentation")
viewer.add_image(volume, name="CT Volume", colormap="gray")
viewer.add_labels(segmentation, name="Segmentation", opacity=0.6, color={1: "red"})
viewer.layers["CT Volume"].scale = spacing  # Aspect ratio correct
napari.run()
```

**Pourquoi Napari ?**
- Standard de l'industrie en imagerie médicale
- Utilisé par les équipes R&D médicales
- Interface professionnelle pour validation clinique
- Permet aux médecins de valider la segmentation interactivement

---

## 🏥 4. Prototype d'Inférence Clinique (Streamlit)

### Application : `app.py`

**Fonctionnalités :**
- ✅ Upload de volumes NIfTI
- ✅ Segmentation automatique
- ✅ Affichage des 3 plans (Axial, Coronal, Sagittal)
- ✅ Slider pour navigation dans les slices
- ✅ Overlay segmentation en rouge
- ✅ Téléchargement des résultats

**Lancement :**
```bash
streamlit run app.py
```

**Interface :**
- Upload de fichier
- Sélection du checkpoint
- Visualisation interactive
- Export des résultats

---

## 📁 Structure des Sorties

```
outputs/
├── loss_curves.png              # Courbes de perte
├── dice_curves.png              # Courbes Dice
├── training_curves.png          # Comparaison train/val
├── hausdorff_curve.png          # Distance Hausdorff (optionnel)
└── inference_test/
    ├── 2d_slices.png            # Slices 2D avec overlay
    ├── overlay.png              # Overlay détaillé
    └── segmentation.nii.gz       # Masque de segmentation
```

---

## ✅ Checklist de Validation

### Graphiques Analytiques
- [x] Training Loss / Validation Loss
- [x] Training Dice / Validation Dice
- [x] Hausdorff Distance (optionnel)
- [x] Export PNG haute résolution (300 DPI)

### Visualisation 2D Médicale
- [x] 3 plans standards (Axial, Coronal, Sagittal)
- [x] Overlay segmentation
- [x] Navigation interactive
- [x] Comparaison avec ground truth

### Visualisation 3D Interactive
- [x] Napari intégré
- [x] Scroll dans 3 axes
- [x] Zoom et pan
- [x] Ajustement contraste/opacité
- [x] Spacing correct

### Prototype Clinique
- [x] Interface Streamlit
- [x] Upload de volumes
- [x] Visualisation interactive
- [x] Export des résultats

---

## 🚀 Utilisation Rapide

### 1. Générer les graphiques de training
```bash
python scripts/plot_training_curves.py --log_path logs/training_history_*.json --output_dir outputs/
```

### 2. Analyser dans le notebook
```bash
jupyter notebook notebooks/03_training_analysis.ipynb
```

### 3. Visualiser les prédictions
```bash
jupyter notebook notebooks/04_inference_visualization.ipynb
```

### 4. Prototype clinique
```bash
streamlit run app.py
```

---

## 📝 Notes Importantes

1. **Napari est le standard** : Utilisé par toutes les équipes R&D en imagerie médicale
2. **3 plans obligatoires** : Axial, Coronal, Sagittal pour validation clinique
3. **Graphiques haute résolution** : 300 DPI pour publications
4. **Overlay transparent** : Segmentation en rouge sur volume en niveaux de gris
5. **Spacing correct** : Respect de l'aspect ratio anatomique dans Napari

---

## 🔧 Dépendances

Toutes les dépendances sont dans `requirements.txt` :
- `matplotlib>=3.7.0` - Graphiques 2D
- `napari>=0.4.0` - Visualisation 3D interactive
- `streamlit>=1.28.0` - Prototype clinique

