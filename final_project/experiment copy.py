'''This is the main experiment file'''

import os, sys 
import copy
import numpy as np
from numpy import linalg as LA

import math

from helper import Indicator, Matrix, Params, iterate_d, Plot_params
#from matrixgen import matrix_generate
from algorithm import Samples, Reconstruct, Hybrid_recover, Nnm_recover
from algorithm import All_entry_nnm, Twophase_nnm
#from reconstruction import nnm_reconstruct, hybrid_reconstruct


method_indicator = Indicator(	data_type 	= "jester",
 								show_hybrid = True,
								show_nnm 	= True, 	#normal nnm method
								show_nnm_e	= True,		#all entry nnm 
								show_2phase = True,		#two phase sampling 
								h_budget 	= True, 	#budget parameter
								h_noise 	= False	
							) 

if method_indicator.data_type == "synthetic":
	mu 		= 5
	sig2 	= 1
	m, n 	= 80, 60
	targ_r 	= 3
	A 		= Matrix( np.random.normal(mu, math.sqrt(sig2), (m,n)) )
	A.rank_r_proj(4) #trunc the rank to be 4

if method_indicator.data_type == "movielens":
	A = Matrix(np.loadtxt('./data/movielens.txt', dtype=float).T)
	targ_r = 10

if method_indicator.data_type == "jester":
	A = Matrix(np.loadtxt('./data/jester.txt', dtype=float).T)
	targ_r = 10
	

param = Params(	indic			= method_indicator,
				is_synthetic 	= method_indicator.data_type, 
				matrix_info 	= A, 
				h_budget 		= method_indicator.h_budget, 
				h_noise 		= method_indicator.h_noise, 
				targ_r 			= targ_r, 
				alpha 			= 0.2
			)	
param.lbd = float(sys.argv[1])
print('lambda is',param.lbd)
d_axis = np.arange(int(param.lplotcoeff*param.targ_r), int(param.maxnum_cols-param.rplotcoeff*targ_r),	param.d_interval )
plts = Plot_params(d_axis)

if param.indic.show_nnm_e:
	plts.set_all_entry(All_entry_nnm(param, plts.length))
		
if param.indic.show_2phase:
	plts.set_phase2(Twophase_nnm(param, plts.length)) 

#param.lbd = lbd
if param.indic.show_hybrid:
	plts.set_hyb_nnm(iterate_d( d_axis, param )) 

plts.plotgraph(param, method_indicator)