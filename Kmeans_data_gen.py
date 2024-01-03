import numpy as np
import matplotlib.pyplot as plt

plt.clf()


# Initialisation

class Cluster : 
    def __init__(self, number, sample_size, mean, std):
        self.number = number
        self.size = sample_size
        self.mean = mean
        self.std = std       

# Dimension des donnees
dim = 2

# Generation des datas
# Cluster 1
mean = 5
std = 0.5
sample_size_1 = 185

l_Cluster = []

cluster_1 = Cluster(0, sample_size_1, mean, std)
l_Cluster.append(cluster_1)

cluster_2 = Cluster(1, sample_size_1, mean, std+5)
l_Cluster.append(cluster_2)
""" 
cluster_3 = Cluster(2, sample_size_1, mean, 0.5)
l_Cluster.append(cluster_3)  """

nb_cluster_data = len(l_Cluster)  # démarre à 0

sample_size = 0
for cluster in l_Cluster :
    sample_size += cluster.size
    
sample = np.zeros([sample_size, 4]) # formalisme : "cluster", "estimated_cluster", "x", "y"
data_number = 0
for cluster in l_Cluster :    
    for data in range(cluster.size):     
        if  cluster == cluster_2 :
            x = np.random.normal(cluster.mean, cluster.std) 
            y = np.random.normal(cluster.mean, cluster.std) 

            while np.sqrt(abs(y - mean)**2+abs(x - mean)**2) < 8: 
                y = np.random.normal(cluster.mean, cluster.std) 
                x = np.random.normal(cluster.mean, cluster.std) 
            sample[data_number] = [cluster.number, np.random.randint(0, nb_cluster_data), x, y]
        else : 
            sample[data_number] = [cluster.number, np.random.randint(0, nb_cluster_data), np.random.normal(cluster.mean, cluster.std), np.random.normal(cluster.mean, cluster.std)]
        data_number += 1


# Résolution
# Hyperparamètre
nb_cluster = 2

# Initialisation des centres en selectionnant des points aléatoirement
center = np.zeros([nb_cluster, dim])  # formalisme : "x", "y"
for point in range(nb_cluster):
    center[point] = sample[np.random.randint(0, sample_size)][2:]



# Visualisation
def display(sample, center, sample_size = sample_size):
    """ Display 2D """
    cluster_color = ["r", "g", "c", "y"]    
    for i, data in enumerate(sample):   
        # Visualisation peu robuste car si le numéro du cluster n'est pas celui du centre de gravité tout est indiqué comme erreur d'affectation     
        """ if int(data[1]) != int(data[0]):    # Erreur sur l'identification du cluster
            plt.scatter(data[2], data[3], c = "black", marker = ".")
        else: """
        plt.scatter(data[2], data[3], c = cluster_color[int(data[1])], marker = "+")
    for i, point in enumerate(center):
        plt.scatter(point[0], point[1], c = cluster_color[i], marker = "o")       

def distance(a, b, dim):
    """ Distance quadratique"""
    tmp = 0
    for i in range(dim):
        tmp += (a[i]-b[i])**2
    tmp = np.sqrt(tmp)
    return tmp  


epsilon = 0.01
var_centre_gravite = epsilon +1 
iteration = 0
while var_centre_gravite > epsilon :

    old_center = np.copy(center)
    
    # Affectation des données aux plus proches centres de gravité
    dist = np.zeros([nb_cluster, 1])
    nb_affectation = np.zeros([nb_cluster, 1])
    tmp_center = np.zeros([nb_cluster, dim])

    for i, data in enumerate(sample):
        for point in range(nb_cluster):
            dist[point] = distance(center[point], data[2:], dim)
            tmp = np.argmin(dist)
            sample[i][1] = tmp
            
            nb_affectation[tmp] = nb_affectation[tmp] + 1
            for j in range(dim):
                tmp_center[tmp][j] = tmp_center[tmp][j] + data[2+j]
    
    # Actualisation des centres de gravité
    for point in range(nb_cluster):
        center[point] = tmp_center[point]/nb_affectation[point]
    
    var_centre_gravite = 0
    for point in range(nb_cluster):
        var_centre_gravite += distance(center[point], old_center[point], dim)
    
    plt.figure()
    display(sample, center)
    plt.savefig("{}".format(iteration)+'.jpg', dpi=200)        
    plt.close()
    
    iteration += 1