import numpy as np
import matplotlib.pyplot as plt
from zmod import zernike_class as zclass
from zmod import zernike_fit as zfit
from astropy.io import fits as pyfits
import healpy as hp
import mhealpy as hmap

from pylab import figure, cm

res = 256
lim = np.log2(res)


grd_path = '/home/alphacentauri/grasp_runs/Retangular_MINUS990_304_0.98GHz_{}pts/'.format(res)
grid_file = zclass.Grd_File(path = grd_path)
grid_file.extract_data(verbose = True)
ref_beam = grid_file.beams[0]
ref_beam.generate_grid(verbose = True)
ref_beam.co = ref_beam.circular_mask(0.05, verbose = True)
#ref_data = np.zeros((len(beam.co), 3))
#ref_data[:,0] = np.nan_to_num(ref_beam.co)
#ref_data[:,1] = ref_beam.thetaphi_grid[:,0]
#ref_data[:,2] = ref_beam.thetaphi_grid[:,1]

def reconstruct_beam(Nps):

	cv_path = '/home/alphacentauri/cross_validation/'
	ret = 'Retangular_MINUS990_304_0.98GHz_'+str(Nps)+'pts'
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

	recon_beam_data = zfit.beam_reconstruction(recon[0],[Nps,Nps],verbose = True)

	cols_mock = np.zeros((Nps**2, 4))
	cols_mock[:,0]=recon_beam_data

	recon_grid = zclass.Grd_File(path = grd_path)
	recon_grid.extract_data(verbose = True)
	recon_beam = zclass.Beam_Data(cols_mock, recon_grid.grid_lims[0], recon_grid.grid_centers[0], recon_grid.Nps[0], recon_grid.frequencies[0])
	recon_beam.generate_grid(verbose = True)
	recon_beam.co = recon_beam.circular_mask(0.05, verbose = True)

	#recon_beam.plot_beam(fig_path = '/home/alphacentauri/cross_validation/reconstructed_beam_'+str(Npoints_zernike[0])+'::'+str(Npoints_reference[0])+'.png', verbose = True)

	return recon_beam


def NRMS(beam, ref_beam):
	
	N_ref = len(ref_beam.valid_data)
	ref_val = np.nan_to_num(ref_beam.valid_data)
	N_rec = len(beam.valid_data)
	rec_val = np.nan_to_num(beam.valid_data)
	diff_val = rec_val-ref_val
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

	return [SDQ, SDPr, SASr]


def deg(Z, order):
	for i in range(0,len(Z),order):
		for j in range(0,len(Z),order):
			med = 0
			for k in range(order):
				for l in range(order):
					med += np.nan_to_num(Z[i+k][j+l])
			med=med/order**2
			for k in range(order):
				for l in range(order):
					Z[i+k][j+l]=med
			
	return Z


def deg_beam(beam,order):
	
	cols_mock = np.zeros((res**2, 4))
	cols_mock[:,0]=beam.co

	deg_beam = zclass.Beam_Data(cols_mock, beam.grid_lims, beam.grid_center, beam.Npoints, beam.frequency) 
	Nps = [res,res]
	data = np.nan_to_num(deg_beam.co.reshape(Nps))
	data_deg = deg(data, order)
	data_deg = data_deg.reshape(res**2)
	deg_beam.co = data_deg
	return deg_beam


def plot(beam, fig_path):
	
	val  = np.nan_to_num(beam.co)
	theta = beam.thetaphi_grid[:,0]
	phi = beam.thetaphi_grid[:,1]

	N = len(val)
	Npoints = [res,res]
	XX = (-theta*np.cos(phi)).reshape(Npoints)
	YY = (theta*np.sin(phi)).reshape(Npoints)
	Z = val.reshape(Npoints)

	fig, ax = plt.subplots()
	c = ax.pcolormesh(XX,YY,20*np.log10(Z), shading='auto')
	cbar = fig.colorbar(c, ax = ax)
	cbar.set_label("Amplitude (dB)")
	ax.set_xlabel("Azimuth (rad)")
	ax.set_ylabel("Elevation (rad)")

	plt.savefig(fig_path)


NRMS_list = []
A = reconstruct_beam(res)
NRMS_list.append(NRMS(A, ref_beam))
plot(A, 'reconstructed_beam_{}pts.png'.format(res))
count = 2
res_list = [str(res)]
while lim>0:
	res_list.append(str(int(res/count)))
	lim-=1
	B = deg_beam(A,count)
	B.generate_grid(verbose=True)
	B.circular_mask(0.05, verbose=True)
	NRMS_list.append(NRMS(B, ref_beam))
	plot(B, 'reconstructed_beam_{}pts[x{}].png'.format(res,count))
	count*=2
print(NRMS_list)
x = np.arange(len(res_list))
fig, ax = plt.subplots()
ax.set_xticks(x,res_list)
ax.text(0.3,20,'azul: SDQ\nvermelho: SDPr+SDQ = NRMS\nverde: SASr+SDPr+SDQ (erro da dist. uniforme)', fontsize = 'medium')
ax.set_title('Componentes de NRMS ao diminuir a resolução de um feixe')
ax.set_xlabel('Número de pixels por lado')
ax.set_ylabel(r'Campo elétrico do sinal ($watt^{\frac{1}{2}}$)')
plt.margins(x=0)

y0,y1,y2 = [],[],[]
for i in range(len(x)):
	y0.append(NRMS_list[i][0])
	y1.append(NRMS_list[i][0]+NRMS_list[i][1])
	y2.append(NRMS_list[i][0]+NRMS_list[i][1]+NRMS_list[i][2])
plt.plot(x,y0,'bo')
plt.plot(x,y1,'ro')
plt.plot(x,y2,'go')
plt.savefig('NRMS [{}x{}].png'.format(res,res))


plot(ref_beam, 'reference_beam_{}pts.png'.format(res))
