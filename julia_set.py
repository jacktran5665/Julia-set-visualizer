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
plt.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.05)

iterations = julia_set(complex(init_c_real, init_c_imag))

im = ax.imshow(iterations, cmap=current_cmap, extent=[-2.5, 2.5, -2.5, 2.5])
title = ax.set_title(f'c = {init_c_real} + {init_c_imag}j', color='#7ecfff', fontsize=20, pad=8)
ax.tick_params(colors='#7ecfff')
ax.set_facecolor('#181828')
ax.grid(False)



import matplotlib.animation as animation

real_c = init_c_real
imag_c = init_c_imag
real_direction = 1  # 1 for increasing, -1 for decreasing
imag_direction = 1
real_speed = 0.01  # Adjust for smoothness and performance
imag_speed = 0.016

def animate(frame):
    global real_c, imag_c, real_direction, imag_direction
    real_c += real_direction * real_speed
    if real_c > 1:
        real_c = 1
        real_direction = -1
    elif real_c < -1:
        real_c = -1
        real_direction = 1
    imag_c += imag_direction * imag_speed
    if imag_c > 1:
        imag_c = 1
        imag_direction = -1
    elif imag_c < -1:
        imag_c = -1
        imag_direction = 1
    c = complex(real_c, imag_c)
    new_iterations = julia_set(c)
    im.set_data(new_iterations)
    title.set_text(f'c = {real_c:.5f} + {imag_c:.5f}j')
    return [im, title]

ani = animation.FuncAnimation(fig, animate, interval=5, blit=False)

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

# full screen
try:
    mng = plt.get_current_fig_manager()
    # For TkAgg
    if hasattr(mng, 'window') and hasattr(mng.window, 'state'):
        mng.window.state('zoomed')
    # For Qt5Agg
    elif hasattr(mng, 'full_screen_toggle'):
        mng.full_screen_toggle()
    # For MacOSX backend
    elif hasattr(mng, 'resize') and hasattr(mng, 'window'):
        mng.resize(*mng.window.maxsize())
except Exception as e:
    print(f"[Info] Could not set full screen: {e}", file=sys.stderr)


# Add a custom legend in the bottom right corner for axis description
fig.text(0.99, 0.06, 'x-axis: Real number         \ny-axis: Imaginary number',
         fontsize=12, color='#7ecfff', ha='right', va='bottom',
         bbox=dict(facecolor='#181828', edgecolor='#7ecfff', boxstyle='square,pad=0.5', alpha=0.85))

plt.show()
