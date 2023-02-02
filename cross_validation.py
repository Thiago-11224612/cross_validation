import numpy as np
import matplotlib.pyplot as plt
from zmod import zernike_class as zclass
from zmod import zernike_fit as zfit
from astropy.io import fits as pyfits
import healpy as hp
import mhealpy as hmap
import os

Npoints_reference = [4,4]
Npoints_zernike = [72,72]


cv_path = '/home/alphacentauri/cross_validation/'
results_path = '/home/alphacentauri/cross_validation/{}_rec/'.format(Npoints_zernike[0])
if not os.path.exists(results_path):
	init = True
	os.makedirs(results_path)
	e = open(results_path+'NRMS.txt','w')
else:
	init = False

ret = 'Retangular_MINUS990_304_0.98GHz_'+str(Npoints_zernike[0])+'pts'
recon_path = cv_path+ret+'/'+ret+'.fits'
recon = []
recon_fits = pyfits.open(recon_path)
for HDU in recon_fits:
	print('\nHDU header:')
	HDU.header
	print('HDU data:')
	print(HDU.data)
	recon.append(HDU.data[0])
recon_fits.close()

grd_path = '/home/alphacentauri/grasp_runs/Retangular_MINUS990_304_0.98GHz_'+str(Npoints_reference[0])+'pts/'
grid_file = zclass.Grd_File(path = grd_path)
grid_file.extract_data(verbose = True)
ref_beam = grid_file.beams[0]
ref_beam.generate_grid(verbose = True)
ref_beam.co = ref_beam.circular_mask(0.05, verbose = True)
ref_beam.plot_beam(fig_path = '/home/alphacentauri/cross_validation/{}_rec/reference_beam_'.format(Npoints_zernike[0])+str(Npoints_zernike[0])+'::'+str(Npoints_reference[0])+'.png', verbose = True)

recon_beam_data = zfit.beam_reconstruction(recon[0],Npoints_reference,verbose = True)

cols_mock = np.zeros((Npoints_reference[0]**2, 4))
cols_mock[:,0]=recon_beam_data

recon_grid = zclass.Grd_File(path = grd_path)
recon_grid.extract_data(verbose = True)
recon_beam = zclass.Beam_Data(cols_mock, recon_grid.grid_lims[0], recon_grid.grid_centers[0], recon_grid.Nps[0], recon_grid.frequencies[0])
recon_beam.generate_grid(verbose = True)
recon_beam.co = recon_beam.circular_mask(0.05, verbose = True)
recon_beam.plot_beam(fig_path = '/home/alphacentauri/cross_validation/{}_rec/reconstructed_beam_'.format(Npoints_zernike[0])+str(Npoints_zernike[0])+'::'+str(Npoints_reference[0])+'.png', verbose = True)

ref_cols = np.zeros((Npoints_reference[0]**2, 4))
ref_cols[:,0] = np.nan_to_num(ref_beam.co)

rec_cols = np.zeros((Npoints_reference[0]**2, 4))
rec_cols[:, 0] = np.nan_to_num(recon_beam.co)

N_ref = len(ref_beam.valid_data)
ref_val = np.nan_to_num(ref_beam.valid_data)

N_rec = len(recon_beam.valid_data)
rec_val = np.nan_to_num(recon_beam.valid_data)


diff_cols = rec_cols-ref_cols

diff_val = rec_val-ref_val


res_grid = zclass.Grd_File(path = grd_path)
res_grid.extract_data(verbose = True)
res_beam = zclass.Beam_Data(diff_cols, res_grid.grid_lims[0], res_grid.grid_centers[0], res_grid.Nps[0], res_grid.frequencies[0])
res_beam.generate_grid(verbose = True)
res_beam.co = res_beam.circular_mask(0.05, verbose = True)
res_beam.plot_beam(fig_path = '/home/alphacentauri/cross_validation/{}_rec/residues_'.format(Npoints_zernike[0])+str(Npoints_zernike[0])+'::'+str(Npoints_reference[0])+'.png', verbose = True)

res_cols = res_beam.valid_data
N_res = len(res_cols)


diff, diff_s, G_med = 0, 0, 0

for i in range(N_ref):
	diff += diff_val[i]
	diff_s+=diff_val[i]**2

for i in range(N_rec):
	G_med += rec_val[i]


G_med = G_med/N_rec
G_diff_s = 0
for i in range(N_ref):
	G_diff_s += (G_med-ref_val[i])**2

SDQ = abs(diff/N_ref)
SDPr = np.sqrt(diff_s/N_ref) - SDQ
SASr = np.sqrt(G_diff_s/N_ref) - SDPr -SDQ
if SASr <= 0:
	SASr = 0

print('NRMS = {} V/m, SDQ = {} V/m, SDPr = {} V/m, SASr = {} V/m'.format(SDQ+SDPr, SDQ, SDPr, SASr))

fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(16,4))
				

theta = recon_beam.thetaphi_grid[:,0]
phi = recon_beam.thetaphi_grid[:,1]
XX = (-theta*np.cos(phi)).reshape(recon_beam.Npoints)
YY = (theta*np.sin(phi)).reshape(recon_beam.Npoints)
axs[0].set_title('Reference Beam')
axs[1].set_title('Reconstructed Beam')
axs[2].set_title('Residues')

min1 = np.min(20*np.log10(abs(ref_val)))
max1 = np.max(20*np.log10(abs(ref_val)))

min2 = np.min(20*np.log10(abs(rec_val)))
max2 = np.max(20*np.log10(abs(rec_val)))

min3 = np.min(20*np.log10(abs(diff_val)))
max3 = np.max(20*np.log10(abs(diff_val)))


c1 = axs[0].pcolormesh(XX,YY,20*np.log10(abs(ref_beam.co.reshape(Npoints_reference))), shading="auto")
c2 = axs[1].pcolormesh(XX,YY,20*np.log10(abs(recon_beam.co.reshape(Npoints_reference))), shading="auto")
c3 = axs[2].pcolormesh(XX,YY,20*np.log10(abs(res_beam.co.reshape(Npoints_reference))), shading="auto")
c1.set_clim(min1,max1)
c2.set_clim(min2,max2)
c3.set_clim(min3,max3)
cbar1 = fig.colorbar(c1, ax=axs[0])
cbar1.set_label("Amplitude (dB)")
cbar2 = fig.colorbar(c2, ax=axs[1])
cbar2.set_label("Amplitude (dB)")
cbar3 = fig.colorbar(c3, ax=axs[2])
cbar3.set_label("Amplitude (dB)")

axs[0].set_xlabel("Azimuth (rad)")
axs[0].set_ylabel("Elevation (rad)")
axs[1].set_xlabel("Azimuth (rad)")
axs[1].set_ylabel("Elevation (rad)")
axs[2].set_xlabel("Azimuth (rad)")
axs[2].set_ylabel("Elevation (rad)")
plt.savefig(results_path+'Reference Beam at [{}x{}] resolution.png'.format(Npoints_reference[0],Npoints_reference[0]))



if init:
	e.write('plotside = {} \tNRMS = {} watt^(1/2),\t SDQ = {} watt^(1/2),\t SDPr = {} watt^(1/2),\t SASr = {} watt^(1/2)\n'.format(Npoints_reference[0],SDQ+SDPr, SDQ, SDPr, SASr))
else:
	e = open(results_path+'NRMS.txt','a')
	e.write('plotside = {} \tNRMS = {} watt^(1/2),\t SDQ = {} watt^(1/2),\t SDPr = {} watt^(1/2),\t SASr = {} watt^(1/2)\n'.format(Npoints_reference[0],SDQ+SDPr, SDQ, SDPr, SASr))
print(N_res)
print(N_ref)
print(N_rec)

salva = pyfits.PrimaryHDU([SDQ, SDPr, SASr])
salva.writeto('/home/alphacentauri/cross_validation/{}_rec/reference_beam_'.format(Npoints_zernike[0])+'NRMS_'+str(Npoints_zernike[0])+'::'+str(Npoints_reference[0])+'.fits', overwrite = True)
