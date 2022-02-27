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
# seeds for the various simplex functions.
# Note that the noise is reproducable if you use the same seed
# ------------------------------------------------------------------------
VALLEY_SEED = int(random.random() * 10000)
VALLEY_SEED = 4345
PLAINS_SEED = VALLEY_SEED
MOUNTAIN_SEED = VALLEY_SEED

# ------------------------------------------------------------------------
# global vars
# ------------------------------------------------------------------------
WIDTH = 40
HEIGHT = 40
PLAIN_XFREQ = 10
PLAIN_YFREQ = 10
MOUNTAIN_XFREQ = 5
MOUNTAIN_YFREQ = 3
VALLEY_XFREQ = 5
VALLEY_YFREQ = 3
MOUNTAIN_SCALE = 10
PLAINS_SCALE = 3
EXPONENT_POWER = 2
CRAGGY_FACTOR = 4
SIGMOID_FACTOR = 4


# ------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------
def main():

    while(True):
        print ("Which noise do you want to look at?")
        print ("1. valleys")
        print ("2. plains noise")
        print ("3. mountain noise")
        print ("4. combined noise")
        print ("")
        print ("5. Modify noise parameters")
        print ("6. exit")
        print ("")
        choice = input ("Choose: ")

        # invalid choice, or we want to exit
        if (not choice.isdigit()): continue
        choice = int(choice)
        if (choice < 1 or choice > 6) : continue
        if (choice == 6) : break

        # set new seeds
        if (choice == 5):
            modify_parameters()
            continue

        # generate and display graph
        (noise_generator, name, seed) = get_noise_fn(choice)
        display(noise_generator, name, seed)

# ------------------------------------------------------------------------
# modify parameters
# ------------------------------------------------------------------------
def modify_parameters():
    print ("\nVALLEY:")
    change_valley_params()
    print("\nPLAIN:")
    change_plain_params()
    print("\nMOUNTAINS:")
    change_mtn_params()

# ------------------------------------------------------------------------
# get the required noise function
# ------------------------------------------------------------------------
def get_noise_fn(choice):
    return {
        1: (valleys_on_off(),"Valleys",f"{VALLEY_SEED}"),
        2: (plains_noise(),"Plains",f"{PLAINS_SEED}"),
        3: (mountains_noise(),"Mountains",f"{MOUNTAIN_SEED}"),
        4: (combined_noise(),"Combined",f"{VALLEY_SEED}, {PLAINS_SEED}, {MOUNTAIN_SEED}"),
    }[choice]

# ------------------------------------------------------------------------
# plot graph
# ------------------------------------------------------------------------
def display (noise_fn,name="",seed=""):
    data = generate(noise_fn)
    data = np.array(data['values']).reshape(WIDTH,HEIGHT)
    fig = go.Figure(data=[go.Surface(z=data,colorscale ='Earth')])
    fig.update_layout(autosize=False,
                scene = dict(zaxis = dict(range=[0,MOUNTAIN_SCALE*3],),),
                  width=800, height=800, title_text=f"{name}")

    txt_valley = f'''
    VALLEYs: seed: {VALLEY_SEED}
    frequency: {VALLEY_XFREQ}, {VALLEY_YFREQ}
    slope: {SIGMOID_FACTOR}'''
    txt_plains = f'''
    PLAINS: seed {PLAINS_SEED}
    frequency: {PLAIN_XFREQ}, {PLAIN_YFREQ}
    scale: {PLAINS_SCALE}'''
    txt_mtns = f'''
    MOUNTAINS: seed: {MOUNTAIN_SEED}
    frequency: {MOUNTAIN_XFREQ}, {MOUNTAIN_YFREQ}
    scale: {MOUNTAIN_SCALE}
    power factor: {EXPONENT_POWER}
    cragginess: {CRAGGY_FACTOR}'''

    fig.add_annotation(text=txt_valley,
                  xref="paper", yref="paper",
                  x=-0.1, y=1.0, showarrow=False)
    fig.add_annotation(text=txt_plains,
                  xref="paper", yref="paper",
                  x=-0.1, y=0.95, showarrow=False)
    fig.add_annotation(text=txt_mtns,
                  xref="paper", yref="paper",
                  x=-0.1, y=0.90, showarrow=False)
    fig.show()


# ------------------------------------------------------------------------
# A simple random on/off to define valleys vs mountains
# returns a function
# ------------------------------------------------------------------------
def valleys_on_off():
    tmp = OpenSimplex(VALLEY_SEED)

    def noise(x, y):
        noise = (tmp.noise2(x*VALLEY_XFREQ/WIDTH, y*VALLEY_YFREQ/HEIGHT) + 1) / 2.0
        return _interpolate(0.0, 1.0, noise)


    return noise

# ------------------------------------------------------------------------
# modify properties that affect the valleys
# ------------------------------------------------------------------------
def change_valley_params():
    global VALLEY_SEED,VALLEY_XFREQ,VALLEY_YFREQ,SIGMOID_FACTOR
    print ()
    print ("Just hit return to use default")
    seed = input(f"random seed [{VALLEY_SEED}] ")
    xfreq = input(f"x frequencey [{VALLEY_XFREQ}] ")
    yfreq = input(f"y frequencey [{VALLEY_YFREQ}] ")
    sig = input(f"slope between valley and not valley [{SIGMOID_FACTOR}] ")
    if xfreq != "": VALLEY_XFREQ = int(xfreq)
    if yfreq != "": VALLEY_YFREQ = int(yfreq)
    if sig != "": SIGMOID_FACTOR = int(sig)
    if seed != "":
        VALLEY_SEED = int(seed)
        MOUNTAIN_SEED = int(seed)
        PLAIN_SEED = int(seed)

# ------------------------------------------------------------------------
# define a function for creating noise that is the root of simplex noise
# returns a function
# ------------------------------------------------------------------------
def plains_noise():

    plain_simplex = OpenSimplex(PLAINS_SEED)
    def noise(x, y):
        value = (plain_simplex.noise2(x*PLAIN_XFREQ/WIDTH, y*PLAIN_YFREQ/HEIGHT) + 1)

        value = (value)**0.25

        if value < 0:
            value = 0

        return value * PLAINS_SCALE

    return noise

# ------------------------------------------------------------------------
# modify properties that affect the plains
# ------------------------------------------------------------------------
def change_plain_params():
    global PLAIN_XFREQ, PLAIN_YFREQ,PLAINS_SCALE,PLAINS_SEED
    print ()
    print ("Just hit return to use default")
    seed = input(f"random seed [{PLAINS_SEED}] ")
    xfreq = input(f"x frequencey [{PLAIN_XFREQ}] ")
    yfreq = input(f"y frequencey [{PLAIN_YFREQ}] ")
    max = input(f"maximum height [{PLAINS_SCALE}]")
    if xfreq != "": PLAIN_XFREQ = int(xfreq)
    if yfreq != "": PLAIN_YFREQ = int(yfreq)
    if max != "": PLAINS_SCALE = int(max)
    if seed != "": PLAIN_SEED = int(seed)

# ------------------------------------------------------------------------
# define a function for creating noise
# just a weirdly stretched simplex noise (x is always double of y)
# returns a function
# ------------------------------------------------------------------------
def mountains_noise():
    mtn_simplex = OpenSimplex(MOUNTAIN_SEED)
    def noise(x, y):
        value1 = (mtn_simplex.noise2(x*MOUNTAIN_XFREQ/WIDTH, y*MOUNTAIN_YFREQ/HEIGHT) + 1) * MOUNTAIN_SCALE
        value1 = value1 ** EXPONENT_POWER / ((1.7*MOUNTAIN_SCALE)**(EXPONENT_POWER-1))

        value2 = (mtn_simplex.noise2(x*MOUNTAIN_XFREQ/WIDTH*CRAGGY_FACTOR,
                y*MOUNTAIN_YFREQ/HEIGHT*CRAGGY_FACTOR) + 1) * MOUNTAIN_SCALE/2
        return value1 + value2

    return noise

# ------------------------------------------------------------------------
# modify properties that affect the mtns
# ------------------------------------------------------------------------
def change_mtn_params():
    global MOUNTAIN_XFREQ,MOUNTAIN_YFREQ,MOUNTAIN_SCALE,MOUNTAIN_SEED
    global CRAGGY_FACTOR, EXPONENT_POWER
    print ()
    print ("Just hit return to use default")
    seed = input(f"random seed [{MOUNTAIN_SEED}] ")
    xfreq = input(f"x frequencey [{MOUNTAIN_XFREQ}] ")
    yfreq = input(f"y frequencey [{MOUNTAIN_YFREQ}] ")
    max = input(f"maximum height [{MOUNTAIN_SCALE}]")
    craggy = input(f"craggy factor [{CRAGGY_FACTOR}]")
    power = input(f"power factor [{EXPONENT_POWER}]")
    if xfreq != "": MOUNTAIN_XFREQ = int(xfreq)
    if yfreq != "": MOUNTAIN_YFREQ = int(yfreq)
    if max != "": MOUNTAIN_SCALE = int(max)
    if craggy != "": CRAGGY_FACTOR = int(craggy)
    if power != "": EXPONENT_POWER = int(power)
    if seed != "": MOUNTAIN_SEED = int(seed)


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
