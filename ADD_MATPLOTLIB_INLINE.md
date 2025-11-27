# Instructions pour afficher automatiquement tous les graphiques

## Solution rapide

Ajoutez cette ligne dans la **première cellule de code** de chaque notebook, juste après les imports et **avant** `sys.path.insert` :

```python
# Enable inline plotting in Jupyter - graphs will display automatically
%matplotlib inline
```

## Notebooks à modifier

### 1. `notebooks/01_data_exploration.ipynb`

**Cellule 1** - Ajoutez après `import napari` (ou après les imports) :

```python
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import napari

# Enable inline plotting in Jupyter - graphs will display automatically
%matplotlib inline

sys.path.insert(0, str(Path().absolute().parent))
```

### 2. `notebooks/02_preprocessing_check.ipynb`

**Cellule 1** - Ajoutez après les imports :

```python
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import napari

# Enable inline plotting in Jupyter - graphs will display automatically
%matplotlib inline

sys.path.insert(0, str(Path().absolute().parent))
```

### 3. `notebooks/03_training_analysis.ipynb`

**Cellule 1** - Ajoutez après `import pandas as pd` :

```python
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Enable inline plotting in Jupyter - graphs will display automatically
%matplotlib inline

sys.path.insert(0, str(Path().absolute().parent))
```

### 4. `notebooks/04_inference_visualization.ipynb`

**Cellule 1** - Ajoutez après `import napari` :

```python
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import nibabel as nib
from typing import Optional, Tuple
import napari

# Enable inline plotting in Jupyter - graphs will display automatically
%matplotlib inline

sys.path.insert(0, str(Path().absolute().parent))
```

## Vérification

Après modification, exécutez la première cellule de chaque notebook. Les graphiques matplotlib s'afficheront automatiquement dans les cellules suivantes.

## Note sur Napari

Les visualisations Napari s'ouvrent automatiquement dans une fenêtre séparée - c'est normal et attendu. Fermez la fenêtre pour continuer l'exécution.

