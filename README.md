# GenAlg — Genetic Algorithm from Scratch

GenAlg es una implementación **desde cero** de un Algoritmo Genético en Python, pensada como un módulo pequeño, claro y extensible dentro de un ecosistema (Grimorio) de proyectos de **Machine Learning, Deep Learning y Optimización**.

---

## Características

- Algoritmo Genético genérico
- Soporte para dominios heterogéneos:
  - Continuos (`tuple`)
  - Discretos con reemplazo (`list`)
  - Permutaciones sin reemplazo (`set`)
- Crossover específico para permutaciones (Ordered Crossover – OX)
- Mutación válida para todos los tipos de dominios
- Reproducibilidad mediante `random_state`
- Early stopping configurable

---

## Ejemplo mínimo de optimización:
```python
import numpy as np
from genalg import GenAlg

def f_eval(ind):
    return -np.sum(ind["x"] ** 2)

params = {
    "x": {"dominio": (-5, 5), "shape": (2,)}
}

ga = GenAlg(
    n_pob=50,
    itera=100,
    random_state=42
)

ga.fit(f_eval, params)

print("Score:", ga.get_score())
print("Params:", ga.get_params())
```

Dentro de la notebook **GenAlg.ipynb** se encuentra otro caso de uso, en donde se ataca el problema del viajero.


##  Licencia
**MIT License**
Libre para usar y modificar.

## Autor
El mago Arcibaldo.
![Ocarina](Arcibaldo.png)






