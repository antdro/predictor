# Python 3.6.0 |Anaconda 4.3.1 (64-bit)|

import pandas as pd
import imageio


def create_gif_for_params(path):
    
    """
    Create .gif file animating optimal parameters search. 
    Each slide is shown with one iteration with speed of 1s per slide.
    File gets saved as params.gif.
    """

    files = ["9.png", "10.png", "11.png", "12.png", "1.png", "2.png", "3.png", "4.png", "5.png"]
    images = []

    for file in files:
        images.append(imageio.imread(path + file))

    kargs = {"duration" : 1, "loop" : 1}
    imageio.mimsave('params.gif', images, 'GIF', **kargs)