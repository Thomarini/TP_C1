# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:00:57 2023

@author: nemo18
"""


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def genere_exemple_dim1(xmin, xmax, nb_ex, sigma):
    x = np.arange(xmin, xmax, (xmax-xmin)/nb_ex)
    y = np.sin(-np.pi + 2*x*np.pi) + np.random.normal(loc=0, scale=sigma, size = x.size)
    return x.reshape(-1, 1), y

def getMSE(x, y, reg):
    return sum(pow(reg.predict(x)-y,2))/x.shape[0]

def plot_model(Xa, Ya, Xt, Yt, reg, nameFig):
    plt.figure()
    Ypred = reg.predict(Xt)
    plt.plot(Xa[:,1], Ya, '*r')
    plt.plot(Xt[:,1], Yt, '-b')
    plt.plot(Xt[:,1], Ypred, '-r')
    plt.grid()
    plt.savefig(nameFig+".jpg",dpi=200)
    plt.close()

def plot_error_profile(L_error_app, L_error_test, nameFig):
    plt.figure()
    plt.plot(range(1, len(L_error_app)+1), L_error_app, '-r')
    plt.plot(range(1, len(L_error_test)+1), L_error_test, '-b')
    plt.grid()
    plt.yscale('log')
    plt.savefig(nameFig+'.jpg', dpi=200)
    plt.close()

def plot_confusion(Xt, Yt, reg, nameFig):
    plt.figure()
    plt.plot(Yt, reg.predict(Xt), '.b')
    plt.plot(Yt, Yt, '-r')
    plt.savefig(nameFig+'.jpg', dpi=200)
    plt.close()

def main(degre_max=10, nb_ex=20, sigma=0.2):
    xmin = 0
    xmax = 1.2
    
    xapp, yapp = genere_exemple_dim1(xmin, xmax, nb_ex, sigma)
    xtest, ytest = genere_exemple_dim1(xmin, xmax, 200, 0)
    
    L_error_app = []
    L_error_test = []
    
    for i in range(1, degre_max+1):
        print("degre = {}".format(i))
        # Transformation des données d'entrées des bases d'app et de test
        poly = PolynomialFeatures(degree=i)
        Xa = poly.fit_transform(xapp)
        Xt = poly.fit_transform(xtest)       
        
        
        # Création du modèle linéaire
        reg = LinearRegression()
        
        # Estimation des erreurs d'apprentissage et de test
        L_error_app.append(getMSE(Xa, yapp, reg.fit(Xa, yapp)))
        L_error_test.append(getMSE(Xt, ytest, reg.fit(Xa, yapp)))
        
        # plot du modèle de degré i
        plot_model(Xa, yapp,Xt, ytest, reg, "Model_%02d" % i)
        plot_confusion(Xt, ytest, reg, "Confusion_%02d" % i)
        
        
    best = np.argmin(L_error_test)+1
    print("Meilleur modele -> degré =", best)
    plot_error_profile(L_error_app, L_error_test, "Profil_Err_App_Test")
    
    # Création du modèle final optimal
    
if __name__ == '__main__':
    main()