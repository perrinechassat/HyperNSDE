import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


"""
Set of functions for visualization of results. 

- plot_static_data() --> affiche un jeu de données statics
- plot_longitudinal_data() --> affiche un jeu de données longitudinales
- plot_true_vs_reconstruction() --> affiche les données initiales et la reconstruction après entrainement du modèle (sur les longitudinals et sur les statics)
- plot_true_reconstruction_generation() --> affiche les données initiales, la reconstruction et la génération après entrainement du modèle (sur les longitudinals et sur les statics)
- plot_latent_variables_longitudinal() --> affiche les variables latentes longitudinal en 2D en fonctions des statics ()
- plot_latent_variables_static() ?

"""


# def plot_static_data():
#     pass


# def plot_longitudinal_data(static_vals, static_types):
#     if static_vals.shape[1] == 2 and ('cat' in static_types and'real' in static_types):
#         # Create colormaps
#         static_types
#         cmap0 = plt.cm.Blues
#         cmap1 = plt.cm.Reds
#         def cmap(real, cat):
#             if cat==0:
#                 return cmap0(real)
#             else:
#                 return cmap1(real)
#     pass


# def plot_true_vs_reconstruction():
#     pass


# def plot_true_reconstruction_generation():
#     pass


# def plot_latent_variables_longitudinal():
#     pass


# def plot_latent_variables_static():
#     pass




# # Create colormaps
# cmap0 = plt.cm.Blues
# cmap1 = plt.cm.Reds
# def cmap(real, cat):
#     if cat==0:
#         return cmap0(real)
#     else:
#         return cmap1(real)

# min_val = min(org_dataset.static_vals[:, 0])
# max_val = max(org_dataset.static_vals[:, 0])
# norm = mcolors.Normalize(vmin=min_val, vmax=max_val)


# def plot_trajectories(ax, time, data, static_vals, title, ylabel, labels=None, legend=False, mask=None):
#     for i in range(data.shape[0]):
#         if mask is not None:
#             ax.plot(time[mask[i, :, 0].bool()], data[i, :][mask[i, :, 0].bool()], color=cmap(norm(static_vals[i, 0]), static_vals[i, 1]))
#         else:
#             ax.plot(time, data[i, :], color=cmap(norm(static_vals[i, 0]), static_vals[i, 1]))
#     ax.set_title(title)
#     ax.set_xlabel('Time')
#     ax.set_ylabel(ylabel)
#     if legend:
#         ax.legend(loc='upper right')
#     ax.grid(True)