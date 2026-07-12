"""
A MontePython implementation of the multi-tracer Cls likelihood class.

Author: Matteo Schulz
With contributions by: Andrea Cozzumbo, Riccardo Murgia
Email: matteo.schulz@gssi.it
"""

import os
import tempfile
import warnings

import numpy as np
import scipy.linalg as la
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import binned_statistic, norm
import dill as pickle
import pandas as pd
from astropy.cosmology import Planck15

from classy import Class
import montepython.io_mp as io_mp
from montepython.likelihood_class import Likelihood

warnings.filterwarnings("ignore")

class GWxHI_fullM(Likelihood):

    def __init__(self, path, data, command_line):
        """
        Initialize the likelihood, loading all data matrices, redshift bins,
        and setting up the configuration arrays based on the chosen network.
        """
        try:
            Likelihood.__init__(self, path, data, command_line)
        except IOError:
            raise io_mp.LikelihoodError("Error initializing Likelihood class.")

        # =========================================================
        # 1. File Naming Logic Based on Parameters
        # =========================================================
        binned_str = '_binned' if self.binned == 'yes' else f'_{self.network}'

        self.cl_file = f'dataset_GWxHI_fullM_{self.bias}_{self.method}_{self.nbins}{binned_str}_Cl.pkl'
        self.Cov_file = f'dataset_GWxHI_fullM_{self.bias}_{self.method}_{self.nbins}{binned_str}_Cov.pkl'
        self.InvCov_file = f'dataset_GWxHI_fullM_{self.bias}_{self.method}_{self.nbins}{binned_str}_InvCov.pkl'
        
        self.redshift_file = f'dataset_GWxHI_fullM_{self.bias}_{self.method}_{self.nbins}_{self.network}_zbin.pkl'
        self.redshift_filehi = f'dataset_GWxHI_fullM_{self.bias}_{self.method}_{self.nbins}_{self.network}_zbinhi.pkl'
        self.redshift_filegw = f'dataset_GWxHI_fullM_{self.bias}_{self.method}_{self.nbins}_{self.network}_zbingw.pkl'
        
        self.zedges_filehi = f'dataset_GWxHI_fullM_{self.bias}_{self.method}_{self.nbins}_{self.network}_zedgeshi.pkl'
        self.zedges_filegw = f'dataset_GWxHI_fullM_{self.bias}_{self.method}_{self.nbins}_{self.network}_zedgesgw.pkl'
        
        self.Ngw_file = f'dataset_GWxHI_fullM_{self.bias}_{self.method}_{self.nbins}_{self.network}_NgwBin.pkl'

        # =========================================================
        # 2. Data Loading Helper
        # =========================================================
        def load_pickle(filename):
            filepath = os.path.join(self.data_directory, filename)
            try:
                with open(filepath, 'rb') as filein:
                    return pickle.load(filein)
            except Exception:
                raise io_mp.LoggedError(f"Could not find {filename} in {self.data_directory}. "
                                        "Please provide the absolute path to the datafiles directory.")

        # Load Data
        dataset_cl = load_pickle(self.cl_file)
        dataset_Cov = load_pickle(self.Cov_file)
        dataset_InvCov = load_pickle(self.InvCov_file)
        
        dataset_redshift = load_pickle(self.redshift_file)
        dataset_redshifthi = load_pickle(self.redshift_filehi)
        dataset_redshiftgw = load_pickle(self.redshift_filegw)
        
        dataset_zedgeshi = load_pickle(self.zedges_filehi)
        dataset_zedgesgw = load_pickle(self.zedges_filegw)
        
        dataset_Ngw_perbin = load_pickle(self.Ngw_file)

        # =========================================================
        # 3. Assign Attributes
        # =========================================================
        self.Cl_obs = dataset_cl
        self.ell_obs = np.array(range(2, 2 + len(self.Cl_obs[0])))

        self.redshift_bins = dataset_redshift
        self.redshift_binshi = dataset_redshifthi
        self.redshift_binsgw = dataset_redshiftgw

        self.z_edgeshi = dataset_zedgeshi
        self.z_edgesgw = dataset_zedgesgw

        self.z_min = 0.001
        self.z_max = 9.0

        self.Cov = dataset_Cov
        self.InvCov = dataset_InvCov
        self.Ngw_perbin = dataset_Ngw_perbin

        # Bins for projection
        self.ell_bins = 30
        self.ell_edges = np.linspace(self.ell_obs.min(), self.ell_obs.max(), self.ell_bins)
        self.ell_centers = 0.5 * (self.ell_edges[1:] + self.ell_edges[:-1])

        # Bias Selection setup
        self.select_biashi = self.bias_hi(self.redshift_binshi)
        self.select_biasgw = self.bias_gw(self.redshift_binsgw)
        self.select_biasAll = np.empty(2 * len(self.redshift_bins))
        
        for i in range(len(self.redshift_bins)):
            self.select_biasAll[i] = self.bias_hi(self.redshift_bins[i])
            self.select_biasAll[len(self.redshift_bins) + i] = self.bias_gw(self.redshift_bins[i])

        # Magnification Bias setup
        self.magnBias_hi = [0.4] * len(self.redshift_binshi)
        self.magnBias_gw = self.magnbias_gw(self.redshift_binsgw)
        self.magnBias = np.empty(2 * len(self.redshift_bins))
        
        for i in range(len(self.redshift_bins)):
            self.magnBias[i] = 0.4
            self.magnBias[len(self.redshift_bins) + i] = self.magnbias_gw(self.redshift_bins[i])

        # Width and Network error setup
        self.width_hi = [0.1] * len(self.redshift_binshi)

        if self.network == 'ET2LCE':
            self.sigma_gw = np.array([0.027, 0.036, 0.041, 0.046, 0.053, 0.056, 0.060, 0.064, 0.072, 0.082])
        elif self.network == 'ET2L':
            self.sigma_gw = np.array([0.046, 0.060, 0.071, 0.079, 0.089, 0.095, 0.098, 0.107, 0.116, 0.124])
        elif self.network == 'ETD':
            self.sigma_gw = np.array([0.058, 0.076, 0.087, 0.095, 0.105, 0.110, 0.115, 0.121, 0.127, 0.135])
        
        self.width_gw = self.sigma_gw * self.redshift_binsgw
        self.width = np.concatenate((self.width_hi, self.width_gw))

        self.zz_tot = np.concatenate((self.redshift_binshi, self.redshift_binsgw))
        self.idx_sort = np.argsort(self.zz_tot)
        self.width = self.width[self.idx_sort]

        # Interpolation setups
        self.zz_dndz = np.arange(0.0, 10.0, 0.001)
        self.dl_lcdm = np.array([Planck15.luminosity_distance(z).value for z in self.zz_dndz])
        self.zz_from_dl = interp1d(self.dl_lcdm, self.zz_dndz, kind='cubic', bounds_error=False, fill_value="extrapolate")

        # =========================================================
        # 4. String formatting for CLASS configuration
        # =========================================================
        def format_arr(arr): return ", ".join(f"{x:.4f}" for x in arr)

        self.zz_string = format_arr(self.redshift_bins)
        self.zz_hi_string = format_arr(self.redshift_binshi)
        self.zz_gw_string = format_arr(self.redshift_binsgw)

        self.select_biashi_string = format_arr(self.select_biashi)
        self.select_biasgw_string = format_arr(self.select_biasgw)
        self.select_biasAll_string = format_arr(self.select_biasAll)

        self.magnBiashi_string = format_arr(self.magnBias_hi)
        self.magnBiasgw_string = format_arr(self.magnBias_gw)
        self.magnBias_string = format_arr(self.magnBias)

        self.width_string_hi = format_arr(self.width_hi)
        self.width_string_gw = format_arr(self.width_gw)
        self.width_string = format_arr(self.width)

    # =========================================================
    # Bias and Density Definitions
    # =========================================================
    def bias_gw(self, z):
        """Returns the GW bias at a given redshift."""
        if self.bias == 'bias':
            a_gw, b_gw, c_gw, d_gw = 0.948, -0.553, 0.996, 1.034
            return (a_gw * np.exp(b_gw * (z**d_gw))) + (z**c_gw)
        return 1.0

    def bias_hi(self, z):
        """Returns the HI bias at a given redshift."""
        if self.bias == 'bias':
            a_hi, b_hi, c_hi = 0.22, 1.47, 0.63
            return (a_hi * ((1 + z)**b_hi)) + c_hi
        return 1.0

    def magnbias_gw(self, z):
        """Returns the magnification bias of GWs."""
        if self.bias == 'bias':
            z_nodes = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0])
            s_nodes = np.array([0.0, 0.08, 0.20, 0.30, 0.37, 0.40])
            spline_model = CubicSpline(z_nodes, s_nodes, bc_type=((2, 0.0), (1, 0.0)))
            return spline_model(z)
        return 0.0

    def dNdz_gw(self, z):
        """Returns the number density distribution of GWs."""
        if self.network == 'ET2LCE':
            A_gw, b_gw, c_gw = 4.129e5, 3.2763, 2.4360
        elif self.network == 'ET2L':
            A_gw, b_gw, c_gw = 3.632e5, 3.1699, 2.3106
        elif self.network == 'ETD':
            A_gw, b_gw, c_gw = 3.243e5, 2.3106, 2.1555
        else:
            A_gw, b_gw, c_gw = 8.25e4, 2.4, 1.71

        return A_gw * (z**b_gw) * np.exp(-c_gw * z)

    def dNdz_hi(self, z, cosmo):
        """Returns the temperature brightness distribution of HI (arxiv:2106.09786)."""
        E_z = cosmo.Hubble(z) / cosmo.Hubble(0)
        Omega_hi = 4 * ((1 + z)**0.6) * (1e-4)
        return 44e-6 * (Omega_hi * cosmo.h() / 2.45 / 1e-4) * (((1 + z)**2) / E_z)

    def rhoEvo_hi(self, z, cosmo):
        """Returns the number density evolution of HI."""
        h = cosmo.h()
        rho_crit0 = 1.87847e-29 * (h**2)
        rho_hi = 4 * ((1 + z)**0.6) * (1e-4) * rho_crit0
        return rho_hi
    
    # =========================================================
    # Likelihood Evaluation
    # =========================================================
    def loglkl(self, cosmo, data):
        """Computes the log-likelihood by generating theory Cls and comparing to data."""
        
        log_likelihood = 0.0
        chi2 = np.zeros(len(self.ell_obs))

        zz = self.redshift_bins
        Nbins = len(zz)

        dl_cosmo = np.array([cosmo.luminosity_distance(z) for z in self.zz_dndz])
        H_cosmo = np.array([cosmo.Hubble(z) for z in self.zz_dndz])

        zz_gw_interp = self.zz_from_dl(dl_cosmo)

        convfactor_dndz_2_dnddl = (((1 + self.zz_dndz) * const.c / 1000. / Planck15.H(self.zz_dndz).value) + (self.dl_lcdm / (1 + self.zz_dndz)))**(-1)
        convfactor_dnddl_2_dndz = (((1 + self.zz_dndz) / H_cosmo) + (dl_cosmo / (1 + self.zz_dndz)))

        dndz_hi = []
        rho_hi = []
        dndz_gw = []

        for i in range(len(self.zz_dndz)):
            dndz_hi.append(self.dNdz_hi(self.zz_dndz[i], cosmo))
            rho_hi.append(self.rhoEvo_hi(self.zz_dndz[i], cosmo))
            dndz_gw.append(self.dNdz_gw(zz_gw_interp[i]))

        dnddl_gw = np.array(dndz_gw) * convfactor_dndz_2_dnddl
        final_dndz_gw = dnddl_gw * convfactor_dnddl_2_dndz
        final_dndz_gw = final_dndz_gw / np.trapz(final_dndz_gw, zz_gw_interp)

        output_hi = np.column_stack((self.zz_dndz, dndz_hi))
        output_rho_hi = np.column_stack((self.zz_dndz, rho_hi))
        output_gw = np.column_stack((zz_gw_interp, final_dndz_gw))

        pid = os.getpid()
        ram_dir = '/dev/shm' if os.path.exists('/dev/shm') else None

        # Create temporary RAM files for CLASS integration
        fd_hi, file_hi = tempfile.mkstemp(prefix=f'selection_hi_{pid}_', suffix='.txt', dir=ram_dir)
        fd_evo, file_evo_hi = tempfile.mkstemp(prefix=f'evo_hi_{pid}_', suffix='.txt', dir=ram_dir)
        fd_gw, file_gw = tempfile.mkstemp(prefix=f'selection_gw_{pid}_', suffix='.txt', dir=ram_dir)

        with os.fdopen(fd_hi, 'w') as f: np.savetxt(f, output_hi, fmt='%.6f')
        with os.fdopen(fd_evo, 'w') as f: np.savetxt(f, output_rho_hi, fmt='%.6f')
        with os.fdopen(fd_gw, 'w') as f: np.savetxt(f, output_gw, fmt='%.6f')

        try:
            # ==========================================
            # Noise Terms Computation
            # ==========================================
            T_sys = 28.0
            B = 20e6
            t_obs = 1.8e7
            N_d = 254.0
            S_area = 20000.0
            S_area_rads = S_area * np.pi*np.pi / 180 / 180
            theta_b = 1.22 * 0.21 * (1 + self.redshift_binshi) / 15.0
            n_pol = 1
            f_sky = 0.5

            K_fg = 6e-7
            A_fg = 0.129
            b_fg = -0.081
            c_fg = 0.581

            if self.network == 'ET2LCE':
                ell2_damp_gw = [52441, 29241, 22201, 17161, 12996, 11236, 9025, 7569, 6084, 4761]
            elif self.network == 'ET2L':
                ell2_damp_gw = [400, 256, 256, 225, 196, 169, 169, 169, 144, 144]
            elif self.network == 'ETD':
                ell2_damp_gw = [49, 36, 49, 36, 49, 36, 36, 36, 36, 36]
            else:
                ell2_damp_gw = [0.] * 10

            beam = np.zeros((len(self.redshift_binshi), self.ell_obs[-2]), dtype=float)
            beam_gw = np.zeros((len(self.redshift_binsgw), self.ell_obs[-2]), dtype=float)

            for i in range(len(self.redshift_binshi)):
                beam[i, :] = np.exp(-self.ell_obs * (self.ell_obs + 1) * (((theta_b[i]) / np.sqrt(16 * np.log(2)))**2))
            
            for i in range(len(self.redshift_binsgw)):
                beam_gw[i, :] = np.exp(-self.ell_obs * (self.ell_obs + 1) / ell2_damp_gw[i])

            Cl_noise_fg = np.zeros((len(self.redshift_binshi), self.ell_obs[-2]), dtype=float)
            Cl_noise_instr = np.zeros((len(self.redshift_binshi), self.ell_obs[-2]), dtype=float)

            for i in range(len(self.redshift_binshi)):
                Cl_noise_fg[i, :] = K_fg * A_fg * np.exp(b_fg * (self.ell_obs**c_fg)) / f_sky
                Cl_noise_instr[i, :] = theta_b[i]*theta_b[i] * (((T_sys / self.dNdz_hi(self.redshift_binshi[i], cosmo) / np.sqrt(n_pol * B * t_obs * N_d)) * np.sqrt(S_area_rads / theta_b[i] / theta_b[i]))**2)

            # ==========================================
            # Cl Theory Auto-Correlation
            # ==========================================
            if self.correlation == 'auto':
                cosmo_params = data.cosmo_arguments.copy()
                params_auto = cosmo_params.copy()

                if self.tracer == 'hi':
                    Nbins_auto = len(self.redshift_binshi)
                    params_auto.update({
                        'selection_multitracing': 'no',
                        'selection_window': 'gaussian',
                        'selection_mean': self.zz_hi_string,
                        'selection_width': self.width_string_hi,
                        'selection_bias': self.select_biashi_string,
                        'selection_magnification_bias': self.magnBiashi_string,
                        'selection_dNdz_1': 'file',
                        'selection_dNdz_filepath_1': file_hi,
                        'selection_dNdzevolution_filepath_1': file_evo_hi,
                        'non_diagonal': len(self.redshift_binshi) - 1,
                        'l_switch_limber_for_nc_local_over_z': 60
                    })

                elif self.tracer == 'gw':
                    Nbins_auto = len(self.redshift_binsgw)
                    params_auto.update({
                        'selection_multitracing': 'no',
                        'selection_window': 'gaussian',
                        'selection_mean': self.zz_gw_string,
                        'selection_width': self.width_string_gw,
                        'selection_bias': self.select_biasgw_string,
                        'selection_magnification_bias': self.magnBiasgw_string,
                        'selection_dNdz_1': 'file',
                        'selection_dNdz_filepath_1': file_gw,
                        'selection_dNdzevolution_filepath_1': file_gw,
                        'non_diagonal': len(self.redshift_binsgw) - 1,
                        'l_switch_limber_for_nc_local_over_z': 60
                    })

                class_auto = Class()
                class_auto.set(params_auto)
                class_auto.compute()

                cl_theory = class_auto.density_cl(lmax=self.ell_obs[-1])
                N_iter_auto = int(0.5 * Nbins_auto * (Nbins_auto + 1))
                cl_auto_th = np.vstack([cl_theory['dd'][i][2:] for i in range(N_iter_auto)])

                if self.method == 'multiCLASS':
                    if self.tracer == 'hi':
                        index = 0
                        cl_hh_observed = []
                        for i in range(len(self.redshift_binshi)):
                            for j in range(i, len(self.redshift_binshi)):
                                if j == i:
                                    cl_hh_observed.append((beam[i, :] * beam[j, :] * cl_auto_th[index]) + Cl_noise_fg[i, :] + Cl_noise_instr[i, :])
                                else:
                                    cl_hh_observed.append((beam[i, :] * beam[j, :] * cl_auto_th[index]) + Cl_noise_fg[i, :])
                                index += 1
                        cl_hh_observed = np.vstack(cl_hh_observed)

                    elif self.tracer == 'gw':
                        index = 0
                        cl_gg_observed = []
                        for i in range(len(self.redshift_binsgw)):
                            for j in range(i, len(self.redshift_binsgw)):
                                if j == i:
                                    cl_gg_observed.append((beam_gw[i, :] * beam_gw[j, :] * cl_auto_th[index]) + (1. / self.Ngw_perbin[i]))
                                else:
                                    cl_gg_observed.append((beam_gw[i, :] * beam_gw[j, :] * cl_auto_th[index]))
                                index += 1
                        cl_gg_observed = np.vstack(cl_gg_observed)

                if self.binned == 'yes':
                    cl_binned_th = []
                    for i in range(N_iter_auto):
                        if self.tracer == 'hi':
                            cl_binned_th.append(binned_statistic(self.ell_obs, cl_hh_observed[i], statistic='mean', bins=self.ell_edges)[0])
                        elif self.tracer == 'gw':
                            cl_binned_th.append(binned_statistic(self.ell_obs, cl_gg_observed[i], statistic='mean', bins=self.ell_edges)[0])
                    cl_binned_th = np.vstack(cl_binned_th)

                class_auto.struct_cleanup()
                class_auto.empty()

                if self.tracer == 'hi':
                    Cl_th_hh = cl_binned_th if self.binned == 'yes' else cl_hh_observed
                    delta_Cl = np.array(Cl_th_hh) - np.array(self.Cl_obs)
                elif self.tracer == 'gw':
                    Cl_th_gg = cl_binned_th if self.binned == 'yes' else cl_gg_observed
                    delta_Cl = np.array(Cl_th_gg) - np.array(self.Cl_obs)

            # ==========================================
            # Cl Theory Cross-Correlation
            # ==========================================
            elif self.correlation == 'cross':

                cosmo_params = data.cosmo_arguments.copy()

                # --- 1. HIxHI ---
                params_hh = cosmo_params.copy()
                params_hh.update({
                    'selection_multitracing': 'no',
                    'selection_window': 'gaussian',
                    'selection_mean': self.zz_hi_string,
                    'selection_width': self.width_string_hi,
                    'selection_bias': self.select_biashi_string,
                    'selection_magnification_bias': self.magnBiashi_string,
                    'selection_dNdz_1': 'file',
                    'selection_dNdz_filepath_1': file_hi,
                    'selection_dNdzevolution_filepath_1': file_evo_hi,
                    'non_diagonal': len(self.redshift_binshi) - 1,
                    'l_switch_limber_for_nc_local_over_z': 60
                })

                class_hh = Class()
                class_hh.set(params_hh)
                class_hh.compute()
                cl_theory_hh = class_hh.density_cl(lmax=self.ell_obs[-1])
                N_iter_hh = 0.5 * len(self.redshift_binshi) * (len(self.redshift_binshi) + 1)
                cl_th_hh = np.vstack([cl_theory_hh['dd'][i][2:] for i in range(int(N_iter_hh))])

                if self.method == 'multiCLASS':
                    index = 0
                    cl_hh_observed = []
                    for i in range(len(self.redshift_binshi)):
                        for j in range(i, len(self.redshift_binshi)):
                            if j == i:
                                cl_hh_observed.append((beam[i, :] * beam[j, :] * cl_th_hh[index]) + Cl_noise_fg[i, :] + Cl_noise_instr[i, :])
                            else:
                                cl_hh_observed.append((beam[i, :] * beam[j, :] * cl_th_hh[index]) + Cl_noise_fg[i, :])
                            index += 1
                    cl_hh_observed = np.vstack(cl_hh_observed)

                if self.binned == 'yes':
                    cl_binned_th_hh = []
                    for i in range(int(N_iter_hh)):
                        cl_binned_th_hh.append(binned_statistic(self.ell_obs, cl_th_hh[i], statistic='mean', bins=self.ell_edges)[0])
                    cl_binned_th_hh = np.vstack(cl_binned_th_hh)

                class_hh.struct_cleanup()
                class_hh.empty()

                # --- 2. GWxHI ---
                params_hg = cosmo_params.copy()
                params_hg.update({
                    'selection_multitracing': 'yes',
                    'selection_window': 'gaussian',
                    'selection_mean': self.zz_string,
                    'selection_width': self.width_string,
                    'selection_bias': self.select_biasAll_string,
                    'selection_magnification_bias': self.magnBias_string,
                    'selection_dNdz_1': 'file',
                    'selection_dNdz_2': 'file',
                    'selection_dNdz_filepath_1': file_hi,
                    'selection_dNdz_filepath_2': file_gw,
                    'selection_dNdzevolution_filepath_1': file_evo_hi,
                    'selection_dNdzevolution_filepath_2': file_gw,
                    'non_diagonal': len(self.redshift_bins) - 1,
                    'l_switch_limber_for_nc_local_over_z': 60
                })

                class_hg = Class()
                class_hg.set(params_hg)
                class_hg.compute()
                cl_theory_hg = class_hg.density_cl(lmax=self.ell_obs[-1])
                N_iter_hg = Nbins * Nbins
                cl_th_hg = np.vstack([cl_theory_hg['dd'][i][2:] for i in range(int(N_iter_hg))])

                map_orig_to_sorted = np.argsort(self.idx_sort)
                real_hi_sorted_indices = map_orig_to_sorted[:len(self.redshift_binshi)]
                real_gw_sorted_indices = map_orig_to_sorted[len(self.redshift_binshi):]

                final_cross_stack = []
                for h in range(len(self.redshift_binshi)):
                    for g in range(len(self.redshift_binsgw)):
                        idx_hi = real_hi_sorted_indices[h]
                        idx_gw = real_gw_sorted_indices[g]
                        raw_stack_idx = idx_hi * len(self.redshift_bins) + idx_gw
                        final_cross_stack.append(cl_th_hg[raw_stack_idx])

                cl_cross_final = np.vstack(final_cross_stack)

                if self.method == 'multiCLASS':
                    index = 0
                    cl_hg_observed = []
                    for i in range(len(self.redshift_binshi)):
                        for j in range(len(self.redshift_binsgw)):
                            cl_hg_observed.append(beam[i, :] * beam_gw[j, :] * cl_cross_final[index])
                            index += 1
                    cl_hg_observed = np.vstack(cl_hg_observed)

                if self.binned == 'yes':
                    cl_binned_th_hg = []
                    for i in range(N_iter_hg):
                        cl_binned_th_hg.append(binned_statistic(self.ell_obs, cl_th_hg[i], statistic='mean', bins=self.ell_edges)[0])
                    cl_binned_th_hg = np.vstack(cl_binned_th_hg)

                class_hg.struct_cleanup()
                class_hg.empty()

                # --- 3. GWxGW ---
                params_gg = cosmo_params.copy()
                params_gg.update({
                    'selection_multitracing': 'no',
                    'selection_window': 'gaussian',
                    'selection_mean': self.zz_gw_string,
                    'selection_width': self.width_string_gw,
                    'selection_bias': self.select_biasgw_string,
                    'selection_magnification_bias': self.magnBiasgw_string,
                    'selection_dNdz_1': 'file',
                    'selection_dNdz_filepath_1': file_gw,
                    'selection_dNdzevolution_filepath_1': file_gw,
                    'non_diagonal': len(self.redshift_binsgw) - 1,
                    'l_switch_limber_for_nc_local_over_z': 60
                })

                class_gg = Class()
                class_gg.set(params_gg)
                class_gg.compute()
                cl_theory_gg = class_gg.density_cl(lmax=self.ell_obs[-1])
                N_iter_gg = 0.5 * len(self.redshift_binsgw) * (len(self.redshift_binsgw) + 1)
                cl_th_gg = np.vstack([cl_theory_gg['dd'][i][2:] for i in range(int(N_iter_gg))])

                if self.method == 'multiCLASS':
                    index = 0
                    cl_gg_observed = []
                    for i in range(len(self.redshift_binsgw)):
                        for j in range(i, len(self.redshift_binsgw)):
                            if j == i:
                                cl_gg_observed.append((beam_gw[i, :] * beam_gw[j, :] * cl_th_gg[index]) + (1. / self.Ngw_perbin[i]))
                            else:
                                cl_gg_observed.append((beam_gw[i, :] * beam_gw[j, :] * cl_th_gg[index]))
                            index += 1
                    cl_gg_observed = np.vstack(cl_gg_observed)

                if self.binned == 'yes':
                    cl_binned_th_gg = []
                    for i in range(int(N_iter_gg)):
                        cl_binned_th_gg.append(binned_statistic(self.ell_obs, cl_th_gg[i], statistic='mean', bins=self.ell_edges)[0])
                    cl_binned_th_gg = np.vstack(cl_binned_th_gg)

                class_gg.struct_cleanup()
                class_gg.empty()

                # --- 4. Combine and Compare ---
                if self.binned == 'yes':
                    Cl_th = np.concatenate((cl_binned_th_hh, cl_binned_th_hg, cl_binned_th_gg))
                else:
                    Cl_th = np.concatenate((cl_hh_observed, cl_hg_observed, cl_gg_observed))

                delta_Cl = np.array(Cl_th) - np.array(self.Cl_obs)

            # ==========================================
            # Likelihood Calculation
            # ==========================================
            for i in range(len(chi2)):
                delta_cl_i = np.array([X[i] for X in delta_Cl])
                inv_cov_i = np.array(self.InvCov[i])
                chi2[i] = np.dot(delta_cl_i, np.dot(inv_cov_i, delta_cl_i))

            chi2_total = np.sum(chi2)
            log_likelihood = -0.5 * chi2_total

            return log_likelihood

        finally:
            # Clean up temporary RAM files safely
            if os.path.exists(file_hi): os.remove(file_hi)
            if os.path.exists(file_evo_hi): os.remove(file_evo_hi)
            if os.path.exists(file_gw): os.remove(file_gw)