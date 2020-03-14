from numpy import genfromtxt
import matplotlib.pyplot as plot

Z = genfromtxt("Z.csv", delimiter=";")
Φ = genfromtxt("Φ.csv", delimiter=";")

F = Z.dot(Φ.T)

plot.imshow(F)
plot.show()
