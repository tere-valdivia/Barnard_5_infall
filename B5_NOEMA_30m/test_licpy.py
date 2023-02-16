import numpy as np
import matplotlib.pyplot as plt
from licpy.lic import runlic
from licpy.plot import grey_save

# test with a uniform x and y field

field_vx = np.ones((50, 50)) * 5
field_vy = np.zeros((50,50))
xx = np.linspace(0, 50, 50)
yy = np.linspace(0, 50, 50)
texture = runlic(field_vx, field_vy, 4)

fig, axlist = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6,6))
axlist[0].quiver(xx, yy, field_vx, field_vy, scale=2)
axlist[1].imshow(texture, cmap='binary', origin='lower')
grey_save('test_image.pdf', texture)

plt.show()