#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import daft

if __name__ == "__main__":

    # Instantiate the PGM.
    pgm = daft.PGM()

    # Hierarchical parameters.
    pgm.add_node("alpha", r"$\alpha$", 0.5, 0)
    pgm.add_node("l0", r"$l_{0}$"    , 0.5, 1)
    
    pgm.add_node("beta", r"$\beta$", 0.5  , 2)
    pgm.add_node("t0", r"$t_{0}$"  , 0.5, 3)

    pgm.add_node("l", r"$l$", 1.5, 0.5)
    pgm.add_node("t", r"$t$", 1.5, 2.5)

    pgm.add_node("yhat", r"$\hat{y}$", 1.5, 1.5)
    
    pgm.add_node("y", r"$y$", 2.5, 1.5, observed=True)
    

    # Add in the edges.
    pgm.add_edge("alpha", "l")
    pgm.add_edge("l0", "l")
    
    pgm.add_edge("beta", "t")
    pgm.add_edge("t0", "t")

    pgm.add_edge("l", "yhat")
    pgm.add_edge("t", "yhat")


    pgm.add_edge("yhat", "y")

    # And a plate.
    pgm.add_plate([1.15, 0., 1.725, 2.775], label=r"$t = 1, \cdots, T$", shift=-0.1)

    # Render and save.
    pgm.render()

    plt.savefig("holt_winters_plat_diagram.pdf")
    plt.close()
