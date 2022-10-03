import warnings
from functools import reduce

import astropy.stats as st
from astropy.visualization import AsymmetricPercentileInterval
from astropy.visualization.mpl_normalize import ImageNormalize
import numpy as np
import photutils as p
from astropy.wcs import WCS
from photutils.utils import calc_total_error
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from astropy.nddata import Cutout2D
from photutils.centroids import centroid_com


def aperture_photometry(data, header, sources, aperture_radius, sky_inner_radius, sky_outer_radius):
    """
    Calculate Aperture Photometry on a list of sources

    Parameters
    ----------
    data:  `np.ndarray`
        A 2D array of pixel values of your data. From a FITS file, you can create this array with
        `fits.getdata`.

    header: `~astropy.fits.Header`
        A FITS Header object. From a FITS file, you can create this object with
        `fits.getheader`.

    sources: `~astropy.table.Table`
        A table of detected sources for the image. Usually, `photutils.DAOStarFinder` would be
        used to create this list.

    aperture_radius: float
        Radius of the target aperture, in pixels

    sky_inner_radius: float
        Radius of the inner aperture that makes up the sky annulus

    sky_outer_radius: float
        Radius of the outer aperture that makes up the sky annulus

    Returns
    -------
    phot_table: `~astropy.table.Table`
        A table of measurements for each source, including instrumental magnitude and error.
    """
    # make apertures around sources, and annuli for sky estimation
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = p.CircularAperture(positions, r=aperture_radius)
    sky_annulus = p.CircularAnnulus(positions,
                                    r_in=sky_inner_radius, r_out=sky_outer_radius)
    annulus_masks = sky_annulus.to_mask(method='center')

    # aperture photometry - calculates total counts in apertures, with errors
    # calc_total_error uses CCD SNR equation - see Lecture 9
    error_arr = calc_total_error(data, 12.0, 1/header['EGAIN'])
    phot_table = p.aperture_photometry(data, apertures, error=error_arr)

    # calculate the Sky background. Because the sky annulus might have other
    # stars inside it, we will take a CLIPPED MEAN of the counts in the annulus
    # to try and reject the contribution from stars.
    bkg_mean = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(data)
        annulus_data_1d = annulus_data[mask.data > 0]
        mean_sigclip, _, _ = st.sigma_clipped_stats(annulus_data_1d)
        bkg_mean.append(mean_sigclip)

    # now we know the mean sky counts. We multiply by ratio of annulus area to
    # target aperture area. This gives expected number of sky counts in target
    # aperture.
    bkg_mean = np.array(bkg_mean)
    phot_table['sky_mean'] = bkg_mean
    phot_table['aper_bkg'] = bkg_mean * apertures.area
    phot_table['aper_sum_bksub'] = phot_table['aperture_sum'] - phot_table['aper_bkg']

    # Tricky bit now! We want to know what Right Ascension and Declination
    # each pixel corresponds to. We will use the information in the FITS header
    # (the so-called "World Coordinate System") to work this out. Don't worry
    # if this code doesn't make sense to you!
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wcs = WCS(header)
    ra, dec = wcs.all_pix2world(phot_table['xcenter'], phot_table['ycenter'], 0)
    phot_table['RA'] = ra
    phot_table['DEC'] = dec

    # Calculate Instrumental Magnitude and Error
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        phot_table['instrumental_mag'] = -2.5*np.log10(phot_table['aper_sum_bksub'] / header['EXPOSURE'])
    phot_table['e_instrumental_mag'] = phot_table['aperture_sum_err'] / phot_table['aper_sum_bksub']

    return phot_table


def plot_sources(data, sources, radius):
    """
    Plot positions of detected sources on image.

    Parameters
    ----------
    data:  `np.ndarray`
        A 2D array of pixel values of your data. From a FITS file, you can create this array with
        `fits.getdata`.

    sources: `~astropy.table.Table`
        A table of detected sources for the image. Usually, `photutils.DAOStarFinder` would be
        used to create this list.

    radius: float
        The size of apertures to plot, in pixels
    """
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = p.CircularAperture(positions, r=radius)
    norm = ImageNormalize(data, interval=AsymmetricPercentileInterval(5, 95))
    fig = plt.figure(figsize=(15, 12))
    plt.imshow(data, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
    apertures.plot(color='red', lw=1.5, alpha=0.5)


def measure_FWHM(data, sources):
    # averagely bright stars
    lims = np.percentile(sources['flux'], (50, 60))
    mask = reduce(
        np.logical_and,
        (sources['flux'] > lims[0], sources['flux'] < lims[1],
         sources['xcentroid'] > 15, sources['ycentroid'] > 15,
         sources['xcentroid'] < data.shape[1]-15, 
         sources['ycentroid'] < data.shape[0]-15
        )
    )

    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    tree = KDTree(positions)
    # nearest neighbours
    dist, ind = tree.query(positions, 2)

    # most isolated star of proper brightness
    idx = dist[:, 1][mask].argmax()
    location = positions[mask][idx]
    cutout = Cutout2D(data, location.T, 15)

    fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))
    norm = ImageNormalize(cutout.data, interval=AsymmetricPercentileInterval(1, 99))
    axis[0].imshow(cutout.data, cmap='Greys', origin='lower', norm=norm, interpolation='nearest')
    axis[0].set_xlabel('X')
    axis[0].set_ylabel('Y')

    xc, yc = centroid_com(cutout.data)
    x = np.arange(15) - xc
    y = np.arange(15) - yc
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    axis[1].plot(R.ravel(), cutout.data.ravel(), '.')
    axis[1].set_xlabel('Distance from centre of star')
    axis[1].set_ylabel('Counts')
