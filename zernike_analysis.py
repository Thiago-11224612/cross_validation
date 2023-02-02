import numpy as np
import sys
import os
import gc
import astropy.io.fits as pyfits
from zmod import zernike_class as zclass
from zmod import zernike_fit as zfit


############################################
############# INPUT DATA ###################
############################################


# Path to directories containing .grd files
grasp_dir = "/home/alphacentauri/resolution_analysis/"
# Directory to contain final results
results_path = "/home/alphacentauri/cross_validation/"
# File name for the final fits containing the coefficients
final_fits = "Retangular_MINUS990_304.fits"
# Figure name for the spectral representation of the Zernike Coefficients
spectral_fig = "spectral_representation.png"
# Minimum Signifficant Coefficient: only plots the coefficients which is above MSC in at least one frequency
msc = 2

# Fitting options
verbose = True
radius   = 0.05 # 0.025
beta_max = 22
pol = "co"
indices     = None # Choose which indices should be fit
show_plot   = True


############################################
########## STARTING ANALYSIS ###############
############################################


print("\n\n1) Executing analysis.\n")
spectral_fig_path = os.path.join(results_path, spectral_fig)
beam_dirs = [beamd for beamd in os.listdir(grasp_dir) if "pts" in beamd]
N = len(beam_dirs)
print("\n{} beam files found:".format(N))
print(beam_dirs)
print()

Coeffs = []
freqs = []
rec_powers = []

for i in range(N):

	beamd = beam_dirs[i]
	title = beamd
	record_file = results_path + title
	filepath = os.path.join(grasp_dir,beamd)
	
	print("\n\n\nAnalysing {}/{}...\nInput File: {}\nResults File: {}\n"
		  "Title: {}\n\n".format(i+1,N,filepath, record_file, title))
		  
	zernike_analysis = zclass.Zernike_Analysis(radius, beta_max, filepath=filepath,
											   verbose=verbose, pol=pol, indices=indices,
											   show_plot=show_plot,record_file=record_file)
	
	Coeffs.append(zernike_analysis.Coeffs[0])
	freqs.append(zernike_analysis.grd_file.frequencies[0])
	rec_powers.append(zernike_analysis.rec_powers[0])
	# Deleting data to avoid overflow
	del zernike_analysis
	gc.collect()


############################################
############ GATHERING DATA ################
############################################


print("\n\n2) Gathering data into one final fits file.\n")

Coeffs = np.array(Coeffs)
rec_powers = np.array(rec_powers)
freqs = np.array(freqs)

for file_name in os.listdir(results_path):
	if ".fits" in file_name: # Getting any fits just to copy the headers
		with pyfits.open(os.path.join(results_path,file_name)) as ffile:
			radius = ffile[0].header["radius"]
			date = ffile[0].header["date"]
		break
		
# Creating FITS file

hdu = pyfits.PrimaryHDU(Coeffs)
hdu.header["ttype1"] = "coefficients"
hdu.header["ttype2"] = "beta"
hdu.header[" ttype3"] = "alpha"
hdu.header.comments["ttype2"] = "radial index"
hdu.header.comments["ttype3"] = "azimuthal index"
hdu.header["radius"] = radius
hdu.header.comments["radius"] = "angular radius (rad)"
hdu.header["date"] = date

hdu_f = pyfits.BinTableHDU.from_columns([pyfits.Column(name="frequencies",
										 format="D",
										 array=freqs)])
hdu_r = pyfits.BinTableHDU.from_columns([pyfits.Column(name="rec_powers",
										 format="D",
										 array=rec_powers)])
hdul = pyfits.HDUList([hdu, hdu_f, hdu_r])
hdul.writeto(os.path.join(results_path,final_fits),output_verify="warn")


print("\n\n3) Plotting the spectral representation of the coefficients.\n")
zfit.spectral_plot(Coeffs, freqs, msc=msc, verbose=True, fig_path=spectral_fig_path)



