import numpy as np
from typing import Dict
from math import radians, sin, cos, sqrt, atan2
import webbrowser
import folium
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import random
import matplotlib.pyplot as plt


start_time = time.time()

# Leer datos 
def leer_datos():
    df_estadios = pd.read_excel('Datos_TFG.xlsx', sheet_name='Estadios Eurocopa 2024', decimal=',') 
    df_formula1 = pd.read_excel('Datos_TFG.xlsx', sheet_name='Circuitos Formula 1 2024', decimal=',')
    df_metro    = pd.read_excel('Datos_TFG.xlsx', sheet_name='Paradas Metro Madrid', decimal=',')
    return df_estadios, df_formula1, df_metro

df_estadios, df_formula1, df_metro = leer_datos()

def calcular_distancias_entre_dos_ciudades(coord1:  pd.Series, coord2:  pd.Series) -> float:
    R = 6371.0  # Radio de la tierra en kilometros
    # Calcular diferencias
    dlon = radians(coord2['Longitud']) - radians(coord1['Longitud'])
    dlat = radians(coord2['Latitud']) - radians(coord1['Latitud'])
    a = sin(dlat/2)**2 + cos(radians(coord1['Latitud'])) * cos(radians(coord2['Latitud'])) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Calcular distancias entre ciudades
n = len(df_estadios)
distancias_ciudades = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        distancias_ciudades[i, j] = calcular_distancias_entre_dos_ciudades(df_estadios.iloc[i], df_estadios.iloc[j])
#print("Distancias ciudades:")
#print(distancias_ciudades)

# Calcular distancias entre circuitos
n = len(df_formula1)
distancias_circuitos = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        distancias_circuitos[i, j] = calcular_distancias_entre_dos_ciudades(df_formula1.iloc[i], df_formula1.iloc[j])
#print("Distancias circuitos:")
#print(distancias_circuitos)

# Calcular distancias entre paradas de metro
n = len(df_metro)
distancias_paradas = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        distancias_paradas[i, j] = calcular_distancias_entre_dos_ciudades(df_metro.iloc[i], df_metro.iloc[j])
#print("Distancias paradas:")
#print(distancias_paradas)




class Grafo:
    def __init__(self, distancias, directed=False):
        self.nodes_len = len(distancias)
        self.roots = {}
        self.distancias = distancias
        self.directed = directed
        self.nodes = list(range(self.nodes_len))  # Nombres de ciudades como índices

    def addEdge(self, a, b, weight):
        if a not in self.roots:
            self.roots[a] = []
        if b not in self.roots:
            self.roots[b] = []

        self.roots[a].append((b, weight))
        if not self.directed:
            self.roots[b].append((a, weight))

    def distance(self, a, b):
        # Verifica que los índices sean válidos
        if a < 0 or a >= self.nodes_len or b < 0 or b >= self.nodes_len:
            raise ValueError(f"Índices fuera de rango: {a}, {b}")
        return self.distancias[a][b]

    def vertices(self):
        return self.nodes

    def getPathCost(self, path, will_return=False):
        cost = 0
        for i in range(len(path) - 1):
            cost += self.distance(path[i], path[i + 1])
        if will_return:  # Solo añadir el costo de regreso si se requiere
            cost += self.distance(path[-1], path[0])
        return cost

    def showGraph(self):
        for key, values in self.roots.items():
            print(f"Ciudad {key}")
            for neighbor, weight in values:
                print(f"conectado a ciudad {neighbor} a distancia de {weight} unidades")


class GeneticAlgorithmTSP:
    def __init__(self, generations=20, population_size=10, tournamentSize=4, mutationRate=0.1, fit_selection_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.tournamentSize = tournamentSize
        self.mutationRate = mutationRate
        self.fit_selection_rate = fit_selection_rate

    def minCostIndex(self, costs):
        return costs.index(min(costs))

    def find_fittest_path(self, graph):
        population = self.randomizeCities(graph.vertices())
        number_of_fits_to_carryover = math.ceil(self.population_size * self.fit_selection_rate)

        if number_of_fits_to_carryover > self.population_size:
            raise ValueError('fitness Rate must be in [0,1].')

        num = 0

        for generation in range(self.generations):
            num += 1
            newPopulation = []
            fitness = self.computeFitness(graph, population)
            fitIndex = self.minCostIndex(fitness)

            if number_of_fits_to_carryover:
                sorted_population = [x for _, x in sorted(zip(fitness, population))]
                fitness = sorted(fitness)
                newPopulation.extend(sorted_population[:number_of_fits_to_carryover])

            while len(newPopulation) < self.population_size:
                parent1 = self.tournamentSelection(graph, population)
                parent2 = self.tournamentSelection(graph, population)
                offspring = self.crossover(parent1, parent2)
                newPopulation.append(self.mutate(offspring))

            population = newPopulation

            if self.converged(population):
                print("Converged", population)
                print('\nConverged to a local minima.', end='')
                break

        return population[fitIndex], fitness[fitIndex], num

    def randomizeCities(self, graph_nodes):
        result = []
        for _ in range(self.population_size):
            cities = graph_nodes[:]
            random.shuffle(cities)
            result.append(cities)
        return result

    def computeFitness(self, graph, population):
        return [graph.getPathCost(path, will_return=True) for path in population]  # Considera el regreso al inicio si necesario

    def tournamentSelection(self, graph, population):
        tournament_contestants = random.sample(population, self.tournamentSize)
        tournament_contestants_fitness = self.computeFitness(graph, tournament_contestants)
        return tournament_contestants[tournament_contestants_fitness.index(min(tournament_contestants_fitness))]

    def crossover(self, parent1, parent2):
        size = len(parent1)
        index_low, index_high = self.computeTwoPointIndexes(size)
        offspring = [None] * size

        offspring[index_low:index_high + 1] = parent1[index_low:index_high + 1]

        current_pos = index_high + 1
        for gene in parent2:
            if gene not in offspring:
                if current_pos >= size:
                    current_pos = 0
                offspring[current_pos] = gene
                current_pos += 1

        return offspring

    def mutate(self, genome):
        if random.random() < self.mutationRate:
            index_low, index_high = self.computeTwoPointIndexes(len(genome))
            return self.swap(index_low, index_high, genome)
        else:
            return genome

    def computeTwoPointIndexes(self, size):
        if size < 2:
            raise ValueError("El tamaño de la lista debe ser al menos 2 para generar dos índices.")
        index_low = random.randint(0, size - 2)
        index_high = random.randint(index_low + 1, size - 1)
        return index_low, index_high

    def swap(self, index_low, index_high, genome):
        genome = genome[:]
        genome[index_low], genome[index_high] = genome[index_high], genome[index_low]
        return genome

    def converged(self, population):
        return all(genome == population[0] for genome in population)


if __name__ == '__main__':
    # Ejemplo de matriz de distancias (reemplaza con tus datos reales)
    distancias = distancias_paradas

    graph = Grafo(distancias)

    ga_tsp = GeneticAlgorithmTSP(generations=500, population_size=150, tournamentSize=50, mutationRate=0.9, fit_selection_rate=0.9)

    mejor_ruta, mejor_coste, total_iteraciones = ga_tsp.find_fittest_path(graph)


   # Mostrar resultados
print("Mejor ruta:", [f"Estadio {i}: {df_metro.iloc[i]['Lugar']}" for i in mejor_ruta])
print("Mejor coste:", round(mejor_coste, 2), "km")
print("Iteraciones: ",total_iteraciones)

# Crear un mapa centrado en la primera ciudad de la lista
m = folium.Map(location=(df_metro.iloc[mejor_ruta[0]]['Latitud'], df_metro.iloc[mejor_ruta[0]]['Longitud']), zoom_start=6)

# Agregar marcadores para cada ciudad
for i, ciudad_idx in enumerate(mejor_ruta):
    row = df_metro.iloc[ciudad_idx]  
    if i == 0 or i == len(mejor_ruta):
        # Marcador de la ciudad inicial de la mejor ruta (color rojo)
        folium.Marker([row['Latitud'], row['Longitud']], popup=row['Lugar'], icon=folium.Icon(color='red')).add_to(m)
    else:
        # Marcadores de las otras ciudades (color azul)
        folium.Marker([row['Latitud'], row['Longitud']], popup=row['Lugar'], icon=folium.Icon(color='blue')).add_to(m)


# Agregar una polilínea para el camino óptimo
path = [df_metro.iloc[i] for i in mejor_ruta] + [df_metro.iloc[mejor_ruta[0]]]  # lista ordenada de ubicaciones de ciudades
folium.PolyLine([(city['Latitud'], city['Longitud']) for city in path], color='red', weight=3).add_to(m)

# Guardar el mapa en un archivo HTML
map_path = 'optimal_path_map.html'
m.save(map_path)

# Abrir el mapa en el navegador web por defecto
webbrowser.open(map_path)

print("--- %s segundos ---" % (time.time() - start_time))
print("------ -------- ----")

