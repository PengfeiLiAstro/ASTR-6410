import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import os.path
import lya_photons as lp
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

def check_file(fname):
	"""Check whether file exsits or not.
	"""
	flag = os.path.exists(fname)
	if flag == False:
		return False
	fp = open(fname, "r")
	line_num = fp.readlines()
	fp.close()
	if len(line_num) < 30000:
		return False
	return True

def load_profile(bins):
	"""Read in my simulation data and calculate surface 
	brightness profiles.
	"""
	DPATH = "/uufs/chpc.utah.edu/common/home/astro/zheng"\
			"/pfli/LyaCGMProfile/SimData/logM=11.0_z=3.0/"
	c0_list = [0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0,
			   1.05, 1.1, 1.15, 1.2]
	xtrain = []
	ytrain = []
	for i in range(len(c0_list)):
		fname = DPATH + "ModNFW_logM=11.0_z=3.0_logGamma=m13"\
						"_logc=%.2f_CEN_param.txt"%c0_list[i]
#				if check_file(photon_name) == False:
#					continue
		gas = lp.LyaGasModel_DST2(fname)
		calc = lp.LyaPhotonCalculator(gas.logM, gas.z, gas.R_sys, 
			   gas.photons, gas.Nph)
		cen_Ltot = gas.cen_Ltot()
		sb = calc.surface_brightness([cen_Ltot], bins, flag_list = [0])
		xtrain.append(np.array([c0_list[i]]))
		ytrain.append(sb[1])
	return np.array(xtrain), np.array(ytrain)

def training_xy():
	"""Load training "points".
	"""
	bins = np.linspace(0, 2, 15)
	xt, yt = load_profile(bins)
	return xt, yt

def kernel_mat(theta, x):
	"""Kernel function.
	"""
	scale = theta[0]
	len_x = len(x)
	xsq_sum = np.sum(x*x, axis = 1)
	sqA = np.full((len_x, len_x), xsq_sum)
	diff_sq = sqA.T + sqA - 2.0*x.dot(x.T)
	K = np.exp(-0.5*diff_sq/scale**2)
	return K

def profile_predict(theta, xt, yt, xp, **kwargs):
	"""Make predictions on mean value with given scale parameter.
	"""
	scale = theta[0]
	xall = np.append(xt, xp, axis = 0)
	len_xall = len(xall)
	len_xt = len(xt)
	kcomb = kernel_mat(theta, xall)
	Kpt = kcomb[len_xt:len_xall, 0:len_xt]
	Ktt = kcomb[0:len_xt, 0:len_xt]
	Ktt_inv = np.linalg.inv(Ktt)
	yp = Kpt.dot(Ktt_inv).dot(yt)
	return yp

def variance_predict(theta, xt, xp):
	"""Make predictions on varaince with given scale parameter.
	"""
	scale = theta[0]
	xall = np.append(xt, xp, axis = 0)
	len_xall = len(xall)
	len_xt = len(xt)
	kcomb = kernel_mat(theta, xall)
	Ktp = kcomb[0:len_xt, len_xt:len_xall]
	Kpt = kcomb[len_xt:len_xall, 0:len_xt]
	Ktt = kcomb[0:len_xt, 0:len_xt]
	Kpp = kcomb[len_xt:len_xall, len_xt:len_xall]
	Ktt_inv = np.linalg.inv(Ktt)

	cov_mat = Kpp - Kpt.dot(Ktt_inv).dot(Ktp)
	var = np.sqrt(cov_mat.diagonal())
	return var

def lnprob_mini(theta, x, y):
	"""Calculate -2 times likelihood.
	"""
	kmat = kernel_mat(theta, x)
	kmat_det = np.linalg.det(kmat)
	if kmat_det < 1e-20:
		return np.inf
	kmat_inv = np.linalg.inv(kmat)

	lnprob = y.T.dot(kmat_inv).dot(y).trace() \
		   + np.log(abs(kmat_det))
	return lnprob

def maximize_lnprob(xtrain, ytrain):
	""" Find scale parameter with maximized likelihood functions.
	"""
	theta_list = np.arange(1e-2, 5, 1e-2)
	lnprob_list = []
	for theta in theta_list:
		lnprob = lnprob_mini([theta], xtrain, ytrain)
		lnprob_list.append(lnprob)

	ind = np.argmin(lnprob_list)
	return [theta_list[ind]]


def training_selection():
	""" Manually select training points. And do necessary preprocessing 
	like taking log values. list pt is the selection.
	"""
	xm_new = []
	ym_new = []
	xm, ym = training_xy()

#	set1	
	pt = [[0.0], [0.4], [0.8], [0.9], [1.0], [1.1], [1.2]]
#	set2
#	pt = [[0.8], [0.9], [1.0], [1.1], [1.2]]
#	set3
#	pt = [[1.0], [1.1], [1.2]]
#	set4
#	pt = [[0.0], [0.4], [0.8], [0.9]]
 
	for i in range(len(xm)):
		if list(xm[i]) in pt:
			# Determine whether to take log values for x or y.
			xm_new.append(xm[i])
#			xm_new.append(np.power(10, xm[i]))
#			ym_new.append(ym[i])
			ym_new.append(np.log10(ym[i]))
	return np.array(xm_new), np.array(ym_new)


def predicting_selection():
	""" Manually select prediction positions. And do necessary preprocessing 
	like taking log values. list pt is the selection.
	"""
	xm_new = []
	ym_new = []
	xm, ym = training_xy()
#	set1
	pt = [[0.2], [0.6], [0.85], [0.95], [1.05], [1.15]]
#	set2
#	pt = [[0.85], [0.95], [1.05], [1.15]]
#	set3
#	pt = [[1.05], [1.15]]
#	set4
#	pt = [[0.2], [0.6], [0.85]]

	for i in range(len(xm)):
		if list(xm[i]) in pt:
			# Determine whether to take log values for x or y.
			xm_new.append(xm[i])
#			xm_new.append(np.power(10, xm[i]))
#			ym_new.append(ym[i])
			ym_new.append(np.log10(ym[i]))
	return np.array(xm_new), np.array(ym_new)

def main_fitting():
	"""Use GPR to predict the mean and variance at given predicting 
	positions.
	"""
	xt, yt = training_selection()
	xp, ym_sub = predicting_selection()

	# Get scale parameter either by hand or by maximize likelihood.
	theta = maximize_lnprob(xt, yt)
#	theta = [0.5]

	bins = np.linspace(0, 2, 15)
	sb_x = 0.5*(bins[1:] + bins[:-1])

	fig, ax = plt.subplots()
	color_list = [plt.cm.RdYlBu_r(p) for p in np.linspace(0, 1, len(xp))]
	yp = profile_predict(theta, xt, yt, xp) # Predict mean.
	yp_var = variance_predict(theta, xt, xp) # Predict variance.

	for i in range(len(xp)):
		label = r"$\log c_{\rm HI}=%.2f$"%(xp[i][0])
		ax.fill_between(sb_x, yp[i] - yp_var[i], yp[i] + yp_var[i], 
						alpha = 0.4, color = color_list[i])
		ax.plot(sb_x, ym_sub[i], c = color_list[i], linestyle = '-',
				label = label)
		ax.plot(sb_x, yp[i], c = color_list[i], linestyle = 'dotted')
	line, = ax.plot([], [], c = 'k', linestyle = 'dotted', label = 'Fitting')
	line, = ax.plot([], [], c = 'k', linestyle = '-', label = 'training')
	ax.legend(fontsize = 10, loc = 'best', frameon = False)
#	ax.set_yscale('log')
	ax.set_ylim([-18.25, -16.25])
	ax.set_yticks([-18, -17.5, -17, -16.5])
	ax.set_xlabel(r"$\rm R\ [arcsec]$", fontsize = 12)
	ax.set_ylabel(r"$\rm \log SB\ [erg\ s^{-1}\ "
				  "cm^{-2}\ arcsec^{-2}]$", fontsize = 12)
	fig.savefig("SB_Fitting.png", dpi = 200)

def main_training():
	""" Go through the whole GPR and make "predictions" at training positions.
	Test whether the fitting function at least pass training points. (Or 
	whether there are numerical issues.
	"""
	xt, yt = training_selection()
	xp = xt

	theta = maximize_lnprob(xt, yt)
#	theta = [0.5]
	print(theta)

	bins = np.linspace(0, 2, 15)
	sb_x = 0.5*(bins[1:] + bins[:-1])

	fig, ax = plt.subplots()
	color_list = [plt.cm.RdYlBu_r(p) for p in np.linspace(0, 1, len(xp))]
	yp = profile_predict(theta, xt, yt, xt)
	for i in range(len(xp)):
		label = r"$\log c_{\rm HI}=%.2f$"%(xp[i][0])
		ax.plot(sb_x, yt[i], c = color_list[i], linestyle = '-',
				label = label)
		ax.plot(sb_x, yp[i], c = 'k', linestyle = 'dotted')
	line, = ax.plot([], [], c = 'k', linestyle = 'dotted', label = 'Fitting')
	line, = ax.plot([], [], c = 'k', linestyle = '-', label = 'training')
	ax.legend(fontsize = 10, loc = 'best', frameon = False)
#	ax.set_yscale('log')
	ax.set_ylim([-18.25, -16.25])
	ax.set_yticks([-18, -17.5, -17, -16.5])
	ax.set_xlabel(r"$\rm R\ [arcsec]$", fontsize = 12)
	ax.set_ylabel(r"$\rm \log SB \ [erg\ s^{-1}\ "
				  "cm^{-2}\ arcsec^{-2}]$", fontsize = 12)
	fig.savefig("SB_Training.png", dpi = 200)


def main_lnprob():
	""" Plot -2lnP as a function of scale factor.
	"""
	xt, yt = training_selection()
	xp, ym_sub = predicting_selection()

	theta_list = np.arange(1e-2, 2, 1e-2)
	lnprob_list = []
	for theta in theta_list:
		lnprob = lnprob_mini([theta], xt, yt)
		lnprob_list.append(lnprob)

	ind = np.argmin(lnprob_list)
	fig, ax = plt.subplots()
	ax.plot(theta_list, lnprob_list, c = 'royalblue')
	ax.set_yscale('log')
	ax.set_xlabel(r"$l$", fontsize = 12)
	ax.set_ylabel(r"$-2\ln\ Prob$", fontsize = 12)
	fig.savefig("lnP.png", dpi = 200)



main_fitting()
main_training()
main_lnprob()
