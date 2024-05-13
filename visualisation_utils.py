import os, numpy, torch

from matplotlib import pyplot as plt
from matplotlib import colors, colormaps

#brain visualization import
from nilearn import regions, datasets, surface, plotting, image, maskers
from nilearn.plotting import plot_roi, plot_stat_map

#absolute paths
MIST_path = '/home/maellef/DataBase/fMRI_parcellations/MIST_parcellation/Parcellations/MIST_ROI.nii.gz'

def voxels_nii(voxel_data, voxel_mask, t_r=1.49):
    voxel_masker = maskers.NiftiMasker(mask_img=voxel_mask, standardize=False, 
                                       detrend=False, t_r=t_r, smoothing_fwhm=8)
    voxel_masker.fit()
    vox_data_nii = voxel_masker.inverse_transform(voxel_data)
    return vox_data_nii

def surface_fig(parcel_data, vmax, threshold=0, cmap='turbo', inflate=True, colorbar=True, no_background=True, symmetric_cbar=True):     
    nii_data = regions.signals_to_img_labels(parcel_data, MIST_path)
    fig, ax = plotting.plot_img_on_surf(nii_data,
                                        views=['lateral', 'medial'], hemispheres=['left', 'right'], inflate=inflate,
                                        vmax=vmax, threshold=threshold, colorbar=colorbar, cmap=cmap, 
                                        symmetric_cbar=symmetric_cbar, cbar_tick_format="%.1f")
    return fig

def voxel_map(voxel_data, voxel_mask, vmax=None, cut_coords=None, tr = 1.49, bg_img=None, cmap = 'cold_hot') : 
    f = plt.Figure()
    data_nii = voxels_nii(voxel_data, voxel_mask, t_r=tr)
    if bg_img is not None : 
        plotting.plot_stat_map(data_nii, bg_img=bg_img, draw_cross=False, vmax=vmax,
                           display_mode='x', cut_coords=[-63, -57, 57, 63], figure=f,
                              black_bg=True, dim = 0, cmap=cmap)
    else :
        plotting.plot_stat_map(data_nii, draw_cross=False, vmax=vmax,
                           display_mode='x', cut_coords=[-63, -57, 57, 63], figure=f)
    return f

def extend_colormap(original_colormap = 'twilight', percent_start = 0.25, percent_finish = 0.25):
    colormap = colormaps[original_colormap]
    nb_colors = colormap.N
    new_colors_range = colormap(numpy.linspace(0,1,nb_colors))

    n_start = round(nb_colors/(1-percent_start)) - nb_colors if percent_start != 0 else 0
    new_color_start = numpy.array([colormap(0)]*n_start).reshape(-1, new_colors_range.shape[1])
    n_finish = round(nb_colors/(1-percent_finish)) - nb_colors if percent_finish != 0 else 0
    new_color_finish = numpy.array([colormap(0)]*n_finish).reshape(-1, new_colors_range.shape[1])

    new_colors_range = numpy.concatenate((new_color_start,new_colors_range,new_color_finish), axis=0)
    new_colormap = colors.ListedColormap(new_colors_range)
    return new_colormap