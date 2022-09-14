# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 12:06:36 2022

@author: TÚLIO
"""
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def table_generator(
    data: t.Union[pd.DataFrame,np.ndarray], name: str, title: str, footnote: str
) -> None:
    """Generate a table and save into the disk.

    Parameters
    ----------
    data : array-like, shape (``n_samples``,``n_features``) 
        The data to generate the table.

    Returns
    -------
    Image: png, jpeg, svg, etc.
        The resulting image given the parameters and data.
    """
    fig_background_color = "skyblue"
    fig_border = "steelblue"

    row_headers = data.index
    cols_headers = data.columns

    cell: np.ndarray = MinMaxScaler().fit_transform(data.values)

    rcolors = plt.cm.BuPu(np.full(len(row_headers),.1))
    ccolors = plt.cm.BuPu(np.full(len(cols_headers),.1))
    cellcolors = plt.cm.summer(cell)

    plt.figure(
        linewidth = 2, 
        edgecolor = fig_border, 
        facecolor = fig_background_color,
        tight_layout = {'pad': 1}
    )
    return
    