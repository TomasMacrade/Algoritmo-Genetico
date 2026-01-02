import numpy as np
import random

class GenAlg():
    """
    Algoritmo Genético para optimización multidominio.
    
    Esta clase permite resolver problemas de optimización que no pueden
    resolverse mediante métodos analíticos o numéricos tradicionales.
    Soporta parámetros definidos como rangos continuos (tuplas), listas de opciones 
    (discretos) y sets de elementos únicos (permutaciones).
    """
    def __init__(self,n_pob = 1000,ratio_selec = 0.2,ratio_cruz = 0.4, ratio_mutac = 0.4,ratio_new=0,p_mutac = 0.3,itera = 100,early_stop = False,early_stop_param = 10,random_state=None):
        """
        Inicializa el algoritmo.

        Parámetros:
        -----------
        n_pob : int
            Tamaño de la población del algoritmo.
        ratio_selec : float
            Proporción de los individuos generados por selección.
        ratio_cruz : float
            Proporción de los individuos generados por cruzamiento.
        ratio_mutac : float
            Proporción de los individuos generados por mutación.
        ratio_new : float
            Proporción de los individuos generados aleatoriamente.
        p_mutac : float
            Probabilidad de que un gen mute.
        itera : int
            Número máximo de generaciones.
        early_stop : bool
            Si es True, el entrenamiento del algoritmo tiene mecanismos de early stop.
        early_stop_param : int 
            Generaciones a esperar antes del paro por early stop.
        random_state : int
            Semilla de aleatoridad del algoritmo.


        """
        if round(ratio_selec + ratio_cruz + ratio_mutac + ratio_new, 2) != 1 :
            raise ValueError("La suma del ratio de selección, cruzamiento, mutación y nuevos individuos debe ser 1.")
        self.n_pob = n_pob
        self.ratio_selec = ratio_selec
        self.ratio_cruz = ratio_cruz
        self.ratio_mutac = ratio_mutac
        self.ratio_new = ratio_new
        self.p_mutac = p_mutac
        self.iter = itera
        self.early_stop_param = early_stop_param
        self.early_stop = early_stop
        self.params = {}
        self.score = None
        self.random_state = random_state
        self._rng_np = np.random.default_rng(random_state)
        self._rng_py = random.Random(random_state)

    def new_pob(self,params_dict,pob):
        """
        Genera una nueva población aleatoria.

        Parámetros:
        -----------
        params_dict : dict
            Configuración de parámetros.
            El diccionario debe contener el nombre de la variable, el dominio de la misma y su shape.
            Los dominios representados con tuplas serán tomados como variables continuas entre los elementos de la tupla y se samplearan siguiendo una distribución uniforme.
            Los dominios representados con listas serán tomados como variables discretas en donde los valores pueden repetirse.
            Los dominios representados con sets serán tomados como variables discretas cuyos valores deben ser permutados. 
            Ejemplo:
            {'x': {'dominio': (0, 1), 'shape': (1,)}, 'ruta': {'dominio': set(range(10))}}
        pob : int 
            Cantidad de individuos a generar.

        Return:
            dict: Diccionario con los parámetros generados como arrays de NumPy.
        """
        parametros = {}
        for param, config in params_dict.items():
            dominio = config['dominio']
            shape = config.get('shape', (1,))
            
            # Parámetros continuos se codifican como una tupla
            if isinstance(dominio, tuple):
                low, high = dominio
                size = (pob,) + shape
                parametros[param] = self._rng_np.uniform(low, high, size=size)
            
            # Parámetros discretos se codifican como una lista
            elif isinstance(dominio, list):
                total_elements = pob * np.prod(shape)
                samples = self._rng_py.choices(dominio, k=total_elements)
                parametros[param] = np.array(samples).reshape((pob,) + shape)
            
            # Parámetros discretos sin reemplazo se codifican como un set
            elif isinstance(dominio, set):
                lista_dominio = list(dominio)
                poblacion_param = []
                for _ in range(pob):
                    poblacion_param.append(self._rng_py.sample(lista_dominio, k=np.prod(shape)))
                parametros[param] = np.array(poblacion_param).reshape((pob,) + shape)
            else:
                raise ValueError(f"El parámetro {param} no tiene un dominio válido (tuple, list o set).")

        return parametros


    def fit(self,f_eval,params_dict,maxim = True,verb = False,verb_iter = 100):
        """
        Entrena el algoritmo para encontrar un conjunto de parámetos óptimo/subóptimo.

        Parámetros:
        -----------
        f_eval : function
            Función de fitness que recibe un dict y devuelve un float.
        params_dict : dict 
            Definición de los dominios de los parámetros.
        maxim : bool 
            True para maximizar la función, False para minimizarla.
        verb : bool
            Si es True, imprime el progreso en consola.
        verb_iter : int 
            Frecuencia de impresión de resultados.
        """
        self.score = None
        self.params = {}
        poblacion = self.new_pob(params_dict,self.n_pob)
        n_selec = max(1, int(self.n_pob * self.ratio_selec))
        n_cruz = int(self.n_pob * self.ratio_cruz)
        n_mut = int(self.n_pob * self.ratio_mutac)
        n_new = self.n_pob - n_selec - n_cruz - n_mut
        if self.early_stop:
            old_score = None
            early_stop_param = 0

        for iteracion in range(self.iter):
            # Evaluación
            scores = []
            for i in range(self.n_pob):
                individuo = {p: poblacion[p][i] for p in poblacion}
                scores.append(f_eval(individuo))
            scores = np.array(scores)

            # Selección
            if maxim:
                indices_ordenados = np.argsort(scores)[::-1]
            else:
                indices_ordenados = np.argsort(scores)
            indices_mejores = indices_ordenados[:n_selec]

            if verb:
                if iteracion% verb_iter==0:
                    print("El score óptimo alcanzado en la iteración {} fué de {}.".format(iteracion+1,round(scores[indices_mejores[0]],2)))
            self.score = scores[indices_mejores[0]]
            self.params = {p: poblacion[p][indices_mejores[0]] for p in poblacion}

            # Voy guardando todo en una nueva población
            nueva_pob = {p: [] for p in params_dict}
            for p in poblacion:
                nueva_pob[p].extend(poblacion[p][indices_mejores].tolist())

            # Cruzamiento
            for _ in range(n_cruz):
                padre1_idx, padre2_idx = self._rng_py.sample(list(indices_mejores), 2)
                for p in params_dict:
                    dominio = params_dict[p]['dominio']
                    if isinstance(dominio, set):
                        #Ordered Crossover (OX) en el caso de sets
                        parent1 = np.array(poblacion[p][padre1_idx]).flatten()
                        parent2 = np.array(poblacion[p][padre2_idx]).flatten()
                        size = len(parent1)
                        a, b = sorted(self._rng_py.sample(range(size), 2))
                        hijo = [None] * size
                        hijo[a:b] = parent1[a:b]
                        p2_filtrado = [item for item in parent2 if item not in hijo]
                        cursor = 0
                        for i in range(size):
                            if hijo[i] is None:
                                hijo[i] = p2_filtrado[cursor]
                                cursor += 1
                        gene = np.array(hijo).reshape(params_dict[p].get('shape', (1,)))
                    else:
                        gene = poblacion[p][padre1_idx] if self._rng_py.random() > 0.5 else poblacion[p][padre2_idx]
                    
                    nueva_pob[p].append(gene)

            # Mutación
            ruido_genetico = self.new_pob(params_dict, n_mut)
            for i in range(n_mut):
                idx_original = self._rng_py.choice(indices_mejores)
                for p in params_dict:
                    dominio = params_dict[p]['dominio']
                    if self._rng_py.random() < self.p_mutac:
                        if isinstance(dominio, set):
                            # Nueva permutación en caso de sets (Swap)
                            gene = np.array(poblacion[p][idx_original]).copy().flatten()
                            idx1, idx2 = self._rng_py.sample(range(len(gene)), 2)
                            gene[idx1], gene[idx2] = gene[idx2], gene[idx1]
                            gen_mutado = gene.reshape(params_dict[p].get('shape', (1,)))
                        else:
                            # Mutación normal (reemplazo aleatorio)
                            gen_mutado = ruido_genetico[p][i]
                    else:
                        gen_mutado = poblacion[p][idx_original]
                    
                    nueva_pob[p].append(gen_mutado)

            # Nuevos individuos para completar lo que falta
            if n_new > 0:
                nuevos = self.new_pob(params_dict, n_new)
                for p in params_dict:
                    nueva_pob[p].extend(nuevos[p])

            for p in params_dict:
                poblacion[p] = np.array(nueva_pob[p])
                

            if self.early_stop:
                if old_score is not None and abs(old_score - self.score) < 1e-7:
                    early_stop_param += 1
                else:
                    early_stop_param = 0
                old_score = self.score
                if early_stop_param>=self.early_stop_param:
                    break

    def get_params(self):
        """
        Devuelve el conjunto óptimo encontrado en el entrenamiento.

        Return:
            dict: Diccionario con los parámetros óptimos/subóptimos encontrados.
        """
        if self.params == {}:
            raise ValueError("Los parámetros óptimos/subóptimos aún no han sido enconctrados. Ejecutar el método fit para entrenar el algoritmo.")
        return self.params
    

    def get_score(self):
        """
        Devuelve el score del conjunto óptimo encontrado en el entrenamiento.

        Return:
            float: El score que corresponde al mejor conjunto de parámetros hallado.
        """
        if self.score == None:
            raise ValueError("Los parámetros óptimos/subóptimos aún no han sido enconctrados. Ejecutar el método fit para entrenar el algoritmo.")
        return self.score
    

            