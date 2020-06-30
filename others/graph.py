
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import math

x_min = -16
x_max = 16.0

mean = 0
std = 3.0

x = np.linspace(x_min, x_max, 100)

y1 = -np.log(scipy.stats.norm.pdf(x,mean,std))

y2 = [math.exp(x1 - 5) + math.exp(-5 - x1) for x1 in x]

y3 = y1 + y2

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.grid()

plt.xlim(x_min,x_max)
plt.ylim(-2,10)

plt.legend(["Base Loss", "Alt Loss", "Custom Loss" ])

plt.xlabel('x')
plt.ylabel('Loss')

plt.savefig("base_loss_vs_custom_loss.png")
plt.show()

x_min = -16
x_max = 16.0

mean = 0
std = 3.0

x = np.linspace(x_min, x_max, 100)

y1 = -np.log(scipy.stats.norm.pdf(x,mean,std))
y2 = -np.log(scipy.stats.norm.pdf(x,mean,std*2))
y3 = -np.log(scipy.stats.norm.pdf(x,mean,std/2))


plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)
plt.grid()

plt.xlim(x_min,x_max)
plt.ylim(-2,10)

plt.legend(["Real distribution", "Too large distribution", "Too tight distribution" ])

plt.xlabel('x')
plt.ylabel('Loss')

plt.savefig("distribution_comp.png")
plt.show()