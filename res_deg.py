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

	return SDQ, SDPr, SASr


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

plot(ref_beam, 'reference_beam_{}pts.png'.format(res))

def res_deg(res,lim):
	NRMS = []
	A = reconstruct_beam(res)
	A_NRMS = NRMS(A, ref_beam)
	print(A_NRMS)
	NRMS.append(A_NRMS)
	plot(A, 'reconstructed_beam_{}pts.png'.format(res))
	count = 2
	list = [res]
	while lim>0:
		list.append(res/count)
		lim-=1
		B = deg_beam(A,count)
		B.generate_grid(verbose=True)
		B.circular_mask(0.05, verbose=True)
		B_NRMS = NRMS(B, ref_beam)
		print(B_NRMS)
		NRMS.append(B_NRMS)
		plot(B, 'reconstructed_beam_{}pts[x{}].png'.format(res,count))
		count*=2
	
	x = np.arange(len(NRMS))
	fig, ax = plt.subplots()
	a = ax.get_xticks().tolist()
	for i in x:
		a[-(i+1)]=list[-(i+1)]
	ax.set_xticklabels(a)
	ax.text(0.3,20,'azul: SDQ\nvermelho: SDPr+SDQ = NRMS\nverde: SASr+SDPr+SDQ (erro da dist. uniforme)', fontsize = 'medium')
	ax.set_title('Componentes de NRMS ao diminuir a resolução de um feixe')
	ax.set_xlabel('Número de pixels por lado')
	ax.set_ylabel('Campo elétrico do sinal ($watt^{\frac{1}{2}}$)')
	plt.margins(x=0)
	plt.plot(x,NRMS[:,0],'bo')
	plt.plot(x,NRMS[:,0]+NRMS[:,1],'ro')
	plt.plot(x,NRMS[:,0]+NRMS[:,1]+NRMS[:,2],'go')
	plt.savefig('NRMS [{}x{}].png'.format(res,res))


A = reconstruct_beam(res)
A_NRMS = NRMS(A, ref_beam)
print(A_NRMS)
AX,AY,AZ = plot(A, 'reconstructed_beam_{}pts.png'.format(res),subplot=True)

B = deg_beam(A,2)
B.generate_grid(verbose=True)
B.circular_mask(0.05, verbose=True)
B_NRMS = NRMS(B, ref_beam)
print(B_NRMS)
BX,BY,BZ = plot(B, 'reconstructed_beam_{}pts[x2].png'.format(res),subplot = True)

C = deg_beam(A,4)
C.generate_grid(verbose=True)
C.circular_mask(0.05, verbose=True)
C_NRMS = NRMS(C, ref_beam)
print(C_NRMS)
CX,CY,CZ = plot(C, 'reconstructed_beam_{}pts[x4].png'.format(res),subplot = True)

D = deg_beam(A,8)
D.generate_grid(verbose=True)
D.circular_mask(0.05, verbose=True)
D_NRMS = NRMS(D, ref_beam)
print(D_NRMS)
DX,DY,DZ = plot(D, 'reconstructed_beam_{}pts[x8].png'.format(res),subplot = True)

E = deg_beam(A,16)
E.generate_grid(verbose=True)
E.circular_mask(0.05, verbose=True)
E_NRMS = NRMS(E, ref_beam)
print(E_NRMS)
EX,EY,EZ = plot(E, 'reconstructed_beam_{}pts[x16].png'.format(res),subplot = True)

F = deg_beam(A,32)
F.generate_grid(verbose=True)
F.circular_mask(0.05, verbose=True)
F_NRMS = NRMS(F, ref_beam)
print(F_NRMS)
FX,FY,FZ = plot(F, 'reconstructed_beam_{}pts[x32].png'.format(res),subplot = True)

#fig, ax = plt.subplots(nrows=6,ncols=2







#x = [0,1,2,3,4,5]
#NRMS = np.zeros((6,3))
#NRMS[0] = A_NRMS
#NRMS[1] = B_NRMS
#NRMS[2] = C_NRMS
#NRMS[3] = D_NRMS
#NRMS[4] = E_NRMS
#NRMS[5] = F_NRMS
#fig, ax = plt.subplots()
#a = ax.get_xticks().tolist()
#a[-6] = 256
#a[-5] = 128
#a[-4] = 64
#a[-3] = 32
#a[-2] = 16
#a[-1] = 8
#ax.set_xticklabels(a)
















ax.text(0.3,20,'azul: SDQ\nvermelho: SDPr+SDQ = NRMS\nverde: SASr+SDPr+SDQ (erro da dist. uniforme)', fontsize = 'medium')
ax.set_title('Componentes de NRMS ao diminuir a resolução de um feixe')
ax.set_xlabel('Número de pixels por lado')
ax.set_ylabel('Campo elétrico do sinal ($watt^{\frac{1}{2}}$)')
plt.margins(x=0)
plt.plot(x,NRMS[:,0],'bo')
plt.plot(x,NRMS[:,0]+NRMS[:,1],'ro')
plt.plot(x,NRMS[:,0]+NRMS[:,1]+NRMS[:,2],'go')
plt.savefig('NRMS.png')
#salva = pyfits.PrimaryHDU(data)
#hdulist = pyfits.HDUList([salva])
#hdulist.writeto('grid_256pts.fits', overwrite = True)
