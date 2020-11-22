import numpy as np
import matplotlib.pyplot as plt

####################
# helper functions #
####################
def gauss2d(x, y, mx, my, sx, sy):
    return 1 - np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

###################
# create a 2D map #
###################

mx = 320; my = 320; sx = 60; sy = 40

x = np.arange(400)
y = np.arange(400)
x_grid, y_grid = np.meshgrid(x, y)
z = gauss2d(x_grid, y_grid, mx, my, sx, sy)

####################
# gradient descent #
####################
def grad(x_, y):

    eps = 1
    gradx = (gauss2d(x_ + eps, y_, mx, my, sx, sy) - gauss2d(x_ - eps, y_, mx, my, sx, sy)) / (2 * eps)
    grady = (gauss2d(x_, y_ + eps, mx, my, sx, sy) - gauss2d(x_, y_ - eps, mx, my, sx, sy)) / (2 * eps)

    return gradx, grady

def gamma(prev_x, prev_y, curr_x, curr_y, prev_grad_x, prev_grad_y, curr_grad_x, curr_grad_y):

    df = np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])
    d_grad = np.array([curr_grad_x, curr_grad_y]) - np.array([prev_grad_x, prev_grad_y])
    norm = np.sum(d_grad * d_grad)
    g = np.abs(np.sum(df*d_grad) / norm)

    return g


def grad_descent(x0, y0):

    # initializations:
    x_track = [x0]
    y_track = [y0]
    curr_x = x0
    curr_y = y0
    curr_grad_x = grad(x0, y0)[0]
    curr_grad_y = grad(x0, y0)[1]
    g = 100000
    precision = 0.0005
    prev_step_size_x = precision + 1
    prev_step_size_y = precision + 1

    while prev_step_size_x > precision or prev_step_size_y > precision:

        prev_x = curr_x
        prev_y = curr_y

        prev_grad_x = curr_grad_x
        prev_grad_y = curr_grad_y

        curr_x += -g * curr_grad_x
        curr_y += -g * curr_grad_y

        curr_grad_x = grad(curr_x, curr_y)[0]
        curr_grad_y = grad(curr_x, curr_y)[1]

        g = gamma(prev_x, prev_y, curr_x, curr_y, prev_grad_x, prev_grad_y, curr_grad_x, curr_grad_y)

        prev_step_size_x = abs(curr_x - prev_x)
        prev_step_size_y = abs(curr_y - prev_y)

        x_track.append(curr_x)
        y_track.append(curr_y)

    xf = curr_x
    yf = curr_y

    return xf, yf, x_track, y_track


xf, yf, x_track, y_track = grad_descent(155, 250)

print("final = ({},{})".format(xf, yf))
plt.pcolormesh(z)
plt.colorbar()
plt.scatter(x_track, y_track, marker='.')