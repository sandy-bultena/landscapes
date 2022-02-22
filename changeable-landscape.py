##########################################################################
# ACKNOWLEDGEMENTS:
# - most of this code is based off of M. McHugh
#   https://github.com/mmchugh/pyconca_noise
#
# - 3d plotting of code is all me (S. Bultena)
##########################################################################

import math
import random
import plotly.graph_objects as go
import numpy as np


from opensimplex import OpenSimplex

# ------------------------------------------------------------------------
# global vars
# ------------------------------------------------------------------------
WIDTH = 40
HEIGHT = 40
PLAIN_XFREQ = 1
PLAIN_YFREQ = 1
MOUNTAIN_XFREQ = 1
MOUNTAIN_YFREQ = 1
VALLEY_XFREQ = 1
VALLEY_YFREQ = 1
MOUNTAIN_SCALE = 10
PLAINS_SCALE = 1
EXPONENT_POWER = 1
CRAGGY_FACTOR = 1
SIGMOID_FACTOR = 1


# ------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------
def main():

    while(True):
        print ("Which noise do you want to look at?")
        print ("1. simplex noise")
        print ("2. valleys")
        print ("3. plains noise")
        print ("4. mountain noise")
        print ("5. combined noise")
        print ("6. exit")
        print ("")
        choice = input ("Choose: ")

        # invalid choice, or we want to exit
        if (not choice.isdigit()): continue
        choice = int(choice)
        if (choice < 1 or choice > 6) : continue
        if (choice == 6) : break

        # get the appropriate noise function.
        noise_fn = get_noise_fn(choice)()

        # generate noise
        data = generate(noise_fn)
        #print(data['values'])
        data = np.array(data['values']).reshape(WIDTH,HEIGHT)
        fig = go.Figure(data=[go.Surface(z=data,colorscale ='Earth')])
        fig.update_layout(autosize=False,
                scene = dict(zaxis = dict(range=[0,MOUNTAIN_SCALE*3],),),
                  width=800, height=800,)

        fig.show()

# ------------------------------------------------------------------------
# get the required noise function
# ------------------------------------------------------------------------
def get_noise_fn(choice):
    return {
        1: simplex_noise,
        2: valleys_on_off,
        3: plains_noise,
        4: mountains_noise,
        5: combined_noise,
    }[choice]


# ------------------------------------------------------------------------
# define a function for creating simplex noise
# returns a function
# ------------------------------------------------------------------------
def simplex_noise():
    open_simplex = OpenSimplex(int(random.random() * 10000))
    def noise(x, y):
        return (open_simplex.noise2(
            x*PLAIN_XFREQ/WIDTH, y*PLAIN_YFREQ/HEIGHT) + 1
        ) * MOUNTAIN_SCALE

    return noise

# ------------------------------------------------------------------------
# A simple random on/off to define valleys vs mountains
# returns a function
# ------------------------------------------------------------------------
def valleys_on_off():
    tmp = OpenSimplex(int(random.random() * 10000))

    def noise(x, y):
        noise = (tmp.noise2(x*VALLEY_XFREQ/WIDTH, y*VALLEY_YFREQ/HEIGHT) + 1) / 2.0
        return _interpolate(0.0, 1.0, noise)

    return noise


# ------------------------------------------------------------------------
# define a function for creating noise that is the root of simplex noise
# returns a function
# ------------------------------------------------------------------------
def plains_noise():

    open_simplex = OpenSimplex(int(random.random() * 10000))
    def noise(x, y):
        value = (open_simplex.noise2(x*PLAIN_XFREQ/WIDTH, y*PLAIN_YFREQ/HEIGHT) + 1)

        value = (value)**0.25

        if value < 0:
            value = 0

        return value * PLAINS_SCALE

    return noise

# ------------------------------------------------------------------------
# define a function for creating noise
# just a weirdly stretched simplex noise (x is always double of y)
# returns a function
# ------------------------------------------------------------------------
def mountains_noise():
    open_simplex = OpenSimplex(int(random.random() * 10000))
    def noise(x, y):
        value1 = (open_simplex.noise2(x*MOUNTAIN_XFREQ/WIDTH, y*MOUNTAIN_YFREQ/HEIGHT) + 1) * MOUNTAIN_SCALE
        value1 = value1 ** EXPONENT_POWER / ((1.7*MOUNTAIN_SCALE)**(EXPONENT_POWER-1))
        value2 = (open_simplex.noise2(x*MOUNTAIN_XFREQ/WIDTH*CRAGGY_FACTOR,
                y*MOUNTAIN_YFREQ/HEIGHT*CRAGGY_FACTOR) + 1) * MOUNTAIN_SCALE/2
        return value1 + value2

    return noise

# ------------------------------------------------------------------------
# creates a function that combines mountains and plains
# ------------------------------------------------------------------------
def combined_noise():
    m_values = mountains_noise()
    p_values = plains_noise()
    weights = valleys_on_off()

    def noise(x, y):
        m = m_values(x, y)
        p = p_values(x, y)
        w = weights(x, y)

        return (m) * w + p * (1 - w)

    return noise

# ------------------------------------------------------------------------
# generates the noise
# - takes in a function
# ------------------------------------------------------------------------
def generate(noise_function):
    values = []

    for x in range(WIDTH):
        for y in range(WIDTH):
            values.append(noise_function(x, y))

    return {
        'width': WIDTH,
        'height': HEIGHT,
        'values': values,
    }

# ------------------------------------------------------------------------
# modifies a value based on a simple curve.
# It zeroes low values, returns high values unmodified, and
# in between, it smooths a bit (gentle transitions from 0 to 1)
# ------------------------------------------------------------------------
def _simple_curve(value):
    start = 0.4
    end = 0.6
    if value < start:
        return 0.0
    if value > end:
        return 1.0
    return (value - start) * (1 / (end - start))

# ------------------------------------------------------------------------
# modifies a value based on a sigmoid curve.
# gentle transitions from 0 to 1
# value should be between -6 and +6
# ------------------------------------------------------------------------
def _sigmoid(value):
    return 1 / (1 + math.exp(-value))

# ------------------------------------------------------------------------
# adds 'a' and 'b' based on the value of 'weight'
# if weight is low, then 'a' will be returned, else 'b' will be returned
# if weight is somewhere in the middle, a reasonable average of 'a' and 'b'
# will be returned.
# ------------------------------------------------------------------------
def _interpolate(a, b, weight):
    # NOTE: weight value is between 0 and 1

    # new_weight = _simple_curve(weight)

    # change weight scale to -6 -> 6
    weight = 2*SIGMOID_FACTOR*weight - SIGMOID_FACTOR
    new_weight = _sigmoid(weight)
    return a * (1 - new_weight) + b * new_weight

# ------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------
if __name__ == '__main__':
    main()
