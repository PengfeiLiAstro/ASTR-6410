import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad

class Cosmo:
	def __init__(self, H0 = 70, OmegaM = 0.28, OmegaL = 0.72, 
				 OmegaB = 0.046):
		self.H0 = H0
		self.OmegaM = OmegaM
		self.OmegaL = OmegaL
		self.OmegaB = OmegaB
		self.const = PhyConst()
	
	def distance_int(self, z):
		H0 = self.H0
		OmegaL = self.OmegaL
		OmegaM = self.OmegaM
		c = self.const.c_cosmo

		return c/H0/np.sqrt(OmegaL + OmegaM*(1.0 + z)**3)
	
	def comoving_distance(self, z):
		dp = quad(self.distance_int, 0.0, z)
		return dp[0]*1e3*self.const.kpc_to_cm

	def angular_distance(self, z):
		dp = self.comoving_distance(z)
		return dp/(1.0 + z)

	def luminosity_distance(self, z):
		dp = self.comoving_distance(z)
		return dp*(1.0 + z)
	
	def R_vir(self, M_vir, z):
		H0 = self.H0
		OmegaL = self.OmegaL
		OmegaM = self.OmegaM
		Grav_cosmo = self.const.Grav_cosmo
		kpc_to_cm = self.const.kpc_to_cm
		Msun = self.const.Msun
		m_H = self.const.m_H
		Delta = 18*np.pi*np.pi

		Hz_coll = H0*np.sqrt(OmegaL + OmegaM*(1 + z)**3)
		rho_crit = 3*Hz_coll**2/8/np.pi/Grav_cosmo\
				   /np.power(1e3*kpc_to_cm, 3)*Msun/m_H # n_H/cm^{-3}
		R_vir = np.power(3*M_vir/4.0/np.pi/Delta
				/(3*Hz_coll*Hz_coll/8/np.pi/Grav_cosmo), 
				1.0/3.0)*1e3*kpc_to_cm 
		return R_vir # cm


class PhyConst:
	def __init__(self):
		self.nu_0 = 2.4660677e15 #s^{-1}
		self.c_cgs = 2.99792458e10 # cm/s
		self.c_cosmo = 2.99792458e5 # km/s
		self.a_nuL = 6.30e-18      # limit cross-section (cm^2)
		self.m_H = 1.672623e-24  # mass of H atom, g
		self.h_plank = 6.6260755e-27 # Planck constant, cm^2 g/s
		self.Msun = 1.9891e33 # mass of sun, g
		self.Grav_cgs = 6.67384e-8   # Gravitational constant (cm^3 g^-1 s^-2)
		self.Grav_cosmo = 4.3009e-9 # Mpc*(km/s)^2/Msun
		self.kpc_to_cm = 3.086e21    # conversion factor from kpc to cm


#dA = 3.410556e27 # cm
class LyaGasModel_DST1:
	def __init__(self, par_fname):
		self.cosmo = Cosmo()
		self.const = PhyConst()
		self.model_param(par_fname)
		self.Nv = 151
		self.photons_header = ['n', 'mu_point', 'mu_local', 'mu_distant', 
							   'r/R_sys', 'z0/r', 'nu_sL-nu_0', 'tau_totol', 
							   'nu_halo']
		self.vel_header = ['r', 'vpec', 'vhub']
		self.ssc_header = ['r', 'f_HI', 'nt_H', 'Ncol_HI', 'emiss_re', 
						   'emiss_cl']
		self.ss_gas()
		self.vel_gas()
		self.lya_photons()


	def model_param(self, par_fname):
		cosmo = self.cosmo
		fmt = []
		fmt_sub = [float for i in range(8)]
		fmt.extend(fmt_sub)
		fmt_sub = [int for i in range(4)]
		fmt.extend(fmt_sub)
		fmt_sub = [float for i in range(11)]
		fmt.extend(fmt_sub)

		fp = open(par_fname, "r")
		lines = fp.readlines()
		names = lines[0].split()
		values = lines[1].split()
#		print(names, values)
#		print(names, values)
		for i in range(len(names)):
			setattr(self, names[i], fmt[i](values[i]))
		self.ssc_fname = lines[2].split()[0]
		self.vel_fname = lines[3].split()[0]
		self.lasimg_fname = lines[4].split()[0]	
		Mvir = np.power(10, self.logM)
		self.R_vir = cosmo.R_vir(Mvir, self.z)
		self.R_sys = self.R_vir*self.scale_fac
		return True

	def ss_gas(self):
		Nr = self.Nr
		names = self.ssc_header
		fname = self.ssc_fname

		fpd = pd.read_csv(fname, names = names, header = None, skiprows = 1, 
						  sep = '\s+', nrows = Nr)
		density_gas = fpd.to_dict(orient = 'list')
		for pname in density_gas:
			density_gas[pname] = np.array(density_gas[pname])
#		print(density_gas['f_HI'])
		density_gas['nHI'] = density_gas['f_HI']*density_gas['nt_H']
#		print(type(density_gas['nHI']))
		self.density_gas = density_gas
		return True

	def vel_gas(self):
		Nv = self.Nv
		names = self.vel_header
		fname = self.vel_fname

		fpd = pd.read_csv(fname, names = names, header = None, skiprows = 1, 
						  sep = '\s+', nrows = Nv)
		vel = fpd.to_dict(orient = "list")
		for pname in vel:
			vel[pname] = np.array(vel[pname])
		self.velocity = vel
		return True

	def lya_photons(self):
		Nph = self.Nph
		flag = self.flag
		fname = self.lasimg_fname
		names = self.photons_header

		fpd = pd.read_csv(fname, names = names, header = None, 
						  sep = '\s+', nrows = Nph)
		photons = fpd.to_dict(orient = "list")
		for pname in photons:
			photons[pname] = np.array(photons[pname])
		self.photons = photons
		self.photons['flag'] = np.full(Nph, flag)
		return True
	
	def cgm_Ltot(self):
		const = self.const
		h_plank = const.h_plank
		nu_0 = const.nu_0
		gas = self.density_gas
		len_gas = self.Nr

		delta_r = (gas['r'][len_gas - 1] - gas['r'][0])/(len_gas - 1)
		Ltot = np.sum(gas['r']**2*gas['emiss_re']\
				   *delta_r*4*np.pi*h_plank*nu_0)
		return Ltot # erg/s
	
	def cen_Ltot(self):
		logM = self.logM
		H0 = self.cosmo.H0
		h = H0/100.0
		SFR = 0.68*np.power(10, logM)*h/1e10
		Ltot = 1e42*SFR
		return Ltot

class LyaGasModel_DST2:
	def __init__(self, par_fname):
		self.cosmo = Cosmo()
		self.const = PhyConst()
		self.model_param(par_fname)
		self.Nv = 151
		self.photons_header = ['n', 'mu_point', 'mu_local', 'mu_distant', 
							   'r/R_sys', 'z0/r', 'nu_sL-nu_0', 'tau_totol', 
							   'nu_halo']
		self.vel_header = ['r', 'vpec', 'vhub']
		self.ssc_header = ['r', 'f_HI', 'nt_H', 'Ncol_HI', 'emiss_re', 
						   'emiss_cl']
		self.ss_gas()
		self.vel_gas()
		self.lya_photons()


	def model_param(self, par_fname):
		cosmo = self.cosmo
		fmt = []
		fmt_sub = [float for i in range(7)]
		fmt.extend(fmt_sub)
		fmt_sub = [int for i in range(4)]
		fmt.extend(fmt_sub)
		fmt_sub = [float for i in range(11)]
		fmt.extend(fmt_sub)

		fp = open(par_fname, "r")
		lines = fp.readlines()
		names = lines[0].split()
		values = lines[1].split()
#		print(names, values)
		for i in range(len(names)):
			setattr(self, names[i], fmt[i](values[i]))
		self.ssc_fname = lines[2].split()[0]
		self.vel_fname = lines[3].split()[0]
		self.lasimg_fname = lines[4].split()[0]	
		Mvir = np.power(10, self.logM)
		self.R_vir = cosmo.R_vir(Mvir, self.z)
		self.R_sys = self.R_vir*self.scale_fac
		return True

	def ss_gas(self):
		Nr = self.Nr
		names = self.ssc_header
		fname = self.ssc_fname

		fpd = pd.read_csv(fname, names = names, header = None, skiprows = 1, 
						  sep = '\s+', nrows = Nr)
		density_gas = fpd.to_dict(orient = 'list')
		for pname in density_gas:
			density_gas[pname] = np.array(density_gas[pname])
#		print(density_gas['f_HI'])
		density_gas['nHI'] = density_gas['f_HI']*density_gas['nt_H']
#		print(type(density_gas['nHI']))
		self.density_gas = density_gas
		return True

	def vel_gas(self):
		Nv = self.Nv
		names = self.vel_header
		fname = self.vel_fname

		fpd = pd.read_csv(fname, names = names, header = None, skiprows = 1, 
						  sep = '\s+', nrows = Nv)
		vel = fpd.to_dict(orient = "list")
		for pname in vel:
			vel[pname] = np.array(vel[pname])
		self.velocity = vel
		return True

	def lya_photons(self):
		Nph = self.Nph
		flag = self.flag
		fname = self.lasimg_fname
		names = self.photons_header

		fpd = pd.read_csv(fname, names = names, header = None, 
						  sep = '\s+', nrows = Nph)
		photons = fpd.to_dict(orient = "list")
		for pname in photons:
			photons[pname] = np.array(photons[pname])
		self.photons = photons
		self.photons['flag'] = np.full(Nph, flag)
		return True
	
	def cgm_Ltot(self):
		const = self.const
		h_plank = const.h_plank
		nu_0 = const.nu_0
		gas = self.density_gas
		len_gas = self.Nr

		delta_r = (gas['r'][len_gas - 1] - gas['r'][0])/(len_gas - 1)
		Ltot = np.sum(gas['r']**2*gas['emiss_re']\
				   *delta_r*4*np.pi*h_plank*nu_0)
		return Ltot # erg/s
	
	def cen_Ltot(self):
		logM = self.logM
		H0 = self.cosmo.H0
		h = H0/100.0
		SFR = 0.68*np.power(10, logM)*h/1e10
		Ltot = 1e42*SFR
		return Ltot




class LyaPhotonCalculator:
	""" Use CGS unit system.
	Basic data structure is dict.
	Key words have to include: nu_sL-nu_0, flag, mu_point
	"""
	def __init__(self, logM, z_red, R_sys, photons = {}, Nph = 0):
		self.logM = logM
		self.z_red = z_red
		self.R_sys = R_sys
		self.photons = photons
		self.Nph = Nph
		self.const = PhyConst()
		self.cosmo = Cosmo()

	def load_photons(self, data, Nph):
		photons = self.photons
		for name in data:
#			print(name, type(self.photons[name]), data[name][1])
			photons[name] = np.append(photons[name], data[name])
#		print(len(photons['n']), type(photons['n']))
#		exit(0)
		self.photons = photons
		self.Nph += Nph
		return True
	
	def wt_proc(self, Ltot_list, wt_list = [1], flag_list = [0], 
				method = 'both', Ang_pj = 5.0, proc = None, **kwargs):
		""" Weight processing for both spectrum and surface_brightness.
		"""
		photons = self.photons
		Nph = self.Nph

		Ltot = np.full(Nph, 0.0)
		wt = np.full(Nph, 0.0)
		ph_num = np.full(Nph, 0.0)
		
		flag = photons['flag']
		for i in range(len(Ltot_list)):
			ph_type = flag==flag_list[i]
			Ltot[ph_type] = Ltot_list[i]
			wt[ph_type] = wt_list[i]
			ph_num[ph_type] = len(ph_type[ph_type == True])

		weights = Ltot/ph_num*wt	
		if method == 'both':
			pass
		elif method == 'red':
			nu_diff = photons['nu_sL-nu_0']
			peak_type = nu_diff<=0.0
			weights[peak_type] = 0.0
		elif method == 'blue':
			nu_diff = photons['nu_sL-nu_0']
			peak_type = nu_diff>0.0
			weights[peak_type] = 0.0
		elif method == 'proj':
			mu_point = photons['mu_point']
			R_sys = self.R_sys
			dA = self.cosmo.angular_distance(self.z_red)
			r_proj = R_sys*np.sqrt(1 - mu_point**2)
			print(R_sys)
			R_proj = dA*(Ang_pj/3600/360*2*np.pi)
#			print("dA=%e, R_proj=%e"%(dA, R_proj))
			proj_type = r_proj>R_proj
			weights[proj_type] = 0.0
		elif method == 'custom':
			if proc == None:
				print("Please customize your function proc()")
				exit(0)
			proc(photons, weights, **kwargs)
		else:
			print("Please select the correct method: red, blue, custom.")
			exit(0)
		return weights

	def surface_brightness(self, Ltot_list, bins, **kwargs):
		photons = self.photons
		z_red = self.z_red
		R_sys = self.R_sys
		Nph = self.Nph
		wt_proc = self.wt_proc
		dA = self.cosmo.angular_distance(z_red)
		
		mu_point = photons['mu_point']
		r_proj = R_sys*np.sqrt(1 - mu_point**2)
		ang = r_proj/dA/2/np.pi*360*3600

		weights = wt_proc(Ltot_list, **kwargs)

		ang_bin = bins # arcsec
		rbin = ang_bin*dA*2*np.pi/360/3600
		rbin_low = rbin[:-1]
		rbin_high = rbin[1:]

		sb_x = 0.5*(ang_bin[1:] + ang_bin[:-1])
		ph_stat = np.histogram(ang, bins = ang_bin, weights = weights)
		sb_y = ph_stat[0]
		sb_y /= 4*np.pi**2*(rbin_high**2 - rbin_low**2)*(1 + z_red)**4
		sb_y *= (2*np.pi)**2/(360*3600)**2

		return [sb_x, sb_y]

	def spectrum(self, Ltot_list, bins, **kwargs):
		# wavelenghth will be in Angstrom.
		photons = self.photons
		z_red = self.z_red
		wt_proc = self.wt_proc
		c = self.const.c_cgs
		nu_0 = self.const.nu_0
		dL = self.cosmo.luminosity_distance(z_red)
	
		wavelen = 1e8*c*(1/(photons['nu_sL-nu_0'] + nu_0) - 1/nu_0)*(1 + z_red)
		weights = wt_proc(Ltot_list, **kwargs)

		wavelen_bin = bins
		wavelen_stat = np.histogram(wavelen, bins = bins, weights = weights)

		wavelen_x = (wavelen_bin[1:] + wavelen_bin[:-1])*0.5
		delta_bin = wavelen_bin[1:] - wavelen_bin[:-1]
		wavelen_y = wavelen_stat[0]
		wavelen_y *= 1.0/4/np.pi/(dL**2)*1e20/delta_bin

		return [wavelen_x, wavelen_y]
