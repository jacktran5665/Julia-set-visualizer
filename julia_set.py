import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numba import njit, prange
import sys

plt.rcParams['toolbar'] = 'none'

# Parameters
dim = 800
max_iter = 80
escape_radius = 2

# grid arrays
x = np.linspace(-1.5, 1.5, dim, dtype=np.float32)
y = np.linspace(-1.5, 1.5, dim, dtype=np.float32)
current_cmap = 'magma'

# Julia set calculation
@njit(parallel=True, fastmath=True)
def julia_set_numba(x, y, c, max_iter, escape_radius):
    dim = x.shape[0]
    iterations = np.zeros((dim, dim), dtype=np.int32)
    for i in prange(dim):
        for j in prange(dim):
            zx = x[j]
            zy = y[i]
            z = complex(zx, zy)
            count = 0
            while abs(z) < escape_radius and count < max_iter:
                z = z * z + c
                count += 1
            iterations[i, j] = count
    return iterations

def julia_set(c):
    return julia_set_numba(x, y, c, max_iter, escape_radius)

# Initial c value
init_c_real = -0.7
init_c_imag = 0.27015

# UI
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(18.1, 9.2), facecolor='#101020')
# Make the grid smaller vertically to avoid slider collision
plt.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.13)

# Initial Julia set
iterations = julia_set(complex(init_c_real, init_c_imag))
# Make the grid smaller in the plot area as well
im = ax.imshow(iterations, cmap=current_cmap, extent=[-2.0, 2.0, -2.0, 2.0])
title = ax.set_title(f'c = {init_c_real} + {init_c_imag}j', color='#7ecfff', fontsize=20, pad=8)
ax.tick_params(colors='#7ecfff')
ax.set_facecolor('#181828')
ax.grid(False)

# Slider axes
ax_real = plt.axes([0.08, 0.01, 0.35, 0.03], facecolor='#222244')
ax_imag = plt.axes([0.53, 0.01, 0.35, 0.03], facecolor='#222244')

# Sliders for real and imaginary parts of c
slider_real = Slider(ax_real, 'X-axis', -1.5, 1.5, valinit=init_c_real, color='#7ecfff')
slider_imag = Slider(ax_imag, 'Y-axis', -1.5, 1.5, valinit=init_c_imag, color='#7ecfff')

# Function to update plot when sliders are changed
def update(val):
    c = complex(slider_real.val, slider_imag.val)
    new_iterations = julia_set(c)
    im.set_data(new_iterations)
    title.set_text(f'c = {slider_real.val:.5f} + {slider_imag.val:.5f}j')
    fig.canvas.draw_idle()

slider_real.on_changed(update)
slider_imag.on_changed(update)

from matplotlib.widgets import RadioButtons

# colormap selection
ax_cmap = plt.axes([0.94, 0.30, 0.05, 0.35], facecolor='#222244')
colormaps = ['magma', 'inferno', 'plasma', 'cividis', 'viridis', 'hot', 'cool', 'hsv']
cmap_selector = RadioButtons(ax_cmap, colormaps, active=colormaps.index(current_cmap))
for label in cmap_selector.labels:
    label.set_color('#7ecfff')
cmap_selector.ax.tick_params(colors='#7ecfff')
def change_cmap(label):
    im.set_cmap(label)
    plt.draw()

cmap_selector.on_clicked(change_cmap)

# Add a custom legend in the bottom right corner for axis description
fig.text(0.99, 0.06, 'X-axis: Real number         \nY-axis: Imaginary number',
         fontsize=12, color='#7ecfff', ha='right', va='bottom',
         bbox=dict(facecolor='#181828', edgecolor='#7ecfff', boxstyle='square,pad=0.5', alpha=0.85))

plt.show()
