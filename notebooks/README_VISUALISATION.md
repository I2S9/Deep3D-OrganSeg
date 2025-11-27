# Guide de Visualisation des Notebooks

## Problème : Aucun graphique n'apparaît

### Solutions

#### 1. Vérifier que les cellules sont exécutées
- Exécutez toutes les cellules dans l'ordre (Menu: Run > Run All Cells)
- Les graphiques matplotlib apparaissent dans la sortie des cellules
- Les visualisations Napari ouvrent une fenêtre séparée

#### 2. Vérifier les imports
Assurez-vous que la première cellule d'imports contient :
```python
import matplotlib.pyplot as plt
import napari  # Pour les notebooks 01, 02, 04
```

#### 3. Vérifier que les données sont chargées
- Les cellules de chargement de données doivent s'exécuter sans erreur
- Vérifiez que les chemins de fichiers sont corrects

#### 4. Types de visualisations disponibles

**Matplotlib (graphiques statiques) :**
- Apparaissent directement dans le notebook après `plt.show()`
- Exemples : histogrammes, courbes, slices 2D

**Napari (visualisation 3D interactive) :**
- Ouvre une fenêtre séparée
- Nécessite `napari.run()` pour démarrer
- Fermez la fenêtre pour continuer l'exécution

#### 5. Commandes utiles

Pour forcer l'affichage dans Jupyter :
```python
%matplotlib inline  # Ajoutez en début de notebook
```

Pour Napari dans Jupyter :
```python
%gui qt  # Active l'interface graphique
```

### Notebooks et leurs visualisations

**01_data_exploration.ipynb :**
- Slices 2D (matplotlib)
- Histogrammes d'intensité (matplotlib)
- Visualisation 3D interactive (Napari)

**02_preprocessing_check.ipynb :**
- Comparaison brut vs prétraité (matplotlib)
- Visualisation 3D interactive (Napari)

**03_training_analysis.ipynb :**
- Courbes Dice (matplotlib)
- Courbes de perte (matplotlib)
- Comparaison training vs validation (matplotlib)

**04_inference_visualization.ipynb :**
- 3 vues avec overlay (matplotlib)
- Visualisation 3D interactive (Napari)
- Comparaison prédiction vs ground truth (Napari)

