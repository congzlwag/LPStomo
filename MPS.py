# -*- coding: utf-8 -*-

from numpy import *
import numpy as np
from numpy.random import rand, seed, randn, randint
from numpy.linalg import norm,eig,solve,svd
from os import mkdir,chdir
import pickle
import os
# import sys
class MPS:
	"""
	Base MPS Class
	Parameter:
		initial_state_type:
			"W"/"dimer"/"cluster" and you will obtain the designated MPS state.
			"random": each entry drawn from uniform distribution U[0,1]
	Attributes:
		id_uncanonical:
			-1: neither right-canon nor left-canon, need canonicalization
			in range(space_size-1): self.matrix[id_uncanonical] is the only one not being cannonical
			space_size-1: left-canonicalized
	"""
	def __init__(self, space_size, initial_state_type, **kwarg):
		self._n = space_size
		self._q = 2
		self.cutoff = 0.01
		self.Dmin = 2
		if 'Dmin' in kwarg:
			self.Dmin = kwarg['Dmin']
		self.Dmax = None

		if initial_state_type == "random" or initial_state_type == "randomforTFIM":
			if "randomInitMaxD" in kwarg:
				self.bondims = randint(self.Dmin, kwarg['randomInitMaxD'],self._n)
			else:
				init_bond_dimension = 2
				self.bondims = [init_bond_dimension] * self._n ## bond[i] connect i i+1
			self.bondims[-1] = 1
			self.matrices = []
			self._q = kwarg['q']
			for i in range(space_size):
				self.matrices.append(randn(self.bondims[i-1], self._q, self.bondims[i]) + randn(self.bondims[i-1], kwarg['q'], self.bondims[i]) * 1.j)
			if initial_state_type == "randomforTFIM":
				self.H = kwarg['H']

		elif initial_state_type == "dimer":
			init_bond_dimension = 3
			self.bondims = [init_bond_dimension] * self._n ## bond[i] connect i i+1
			self.bondims[-1] = 1
			self.matrices = []
			
			l = zeros((1,2,3), dtype=complex)
			l[0,1,1] = 1
			l[0,0,2] = 1.
			r = zeros((3,2,1), dtype=complex)
			r[1,0,0] = -1.
			r[2,1,0] = 1.
			bulk = zeros((3,2,3), dtype=complex)
			bulk[0,1,1] = 1.
			bulk[0,0,2] = 1.
			bulk[1,0,0] = -1.
			bulk[2,1,0] = 1.
			self.matrices.append(l.copy())
			for i in range(space_size - 2):
				self.matrices.append(bulk.copy())
			self.matrices.append(r.copy())

		elif initial_state_type == "W":
			init_bond_dimension = 2
			self.bondims = [init_bond_dimension] * self._n ## bond[i] connect i i+1
			self.bondims[-1] = 1
			self.matrices = []
		
			l = zeros((1,2,2), dtype=complex)
			l[0,0,0] = 1
			l[0,1,1] = 1.
			r = zeros((2,2,1), dtype=complex)
			r[1,0,0] = 1.
			r[0,1,0] = exp(0.1j * (self._n - 1))
			bulk = zeros((2,2,2), dtype=complex)
			bulk[0,0,0] = 1.
			bulk[1,0,1] = 1.
			
			self.matrices.append(l.copy())
			for i in range(1, space_size - 1):
				bulk[0,1,1] = exp(0.1j * i)
				self.matrices.append(bulk.copy())
			self.matrices.append(r.copy())
		elif initial_state_type == "Cluster":
			init_bond_dimension = 4
			self.bondims = [init_bond_dimension] * self._n ## bond[i] connect i i+1
			self.bondims[-1] = 1
			self.bondims[0] = 2
			self.bondims[-2] = 2
			
			self.matrices = []
		
			l1 = zeros((1,2,2)) + 0.j
			l1[0,0,0] = 1.
			l1[0,0,1] = -1.
			l1 /= 2 ** (1. / 3.)
			
			l2 = zeros((2,2,4)) + 0.j
			l2[0,0,0] = 1.
			l2[0,0,1] = -1.
			
			l2[1,1,2] = 1.
			l2[1,1,3] = 1.
			
			l2 /= 2 ** (2. / 3.)
			
			bulk = zeros((4,2,4)) + 0.j
			bulk[0,0,0] = 1.
			bulk[0,0,1] = -1.
			bulk[1,1,2] = 1.
			bulk[1,1,3] = 1.
			
			bulk[2,0,0] = -1.
			bulk[2,0,1] = 1.
			bulk[3,1,2] = -1.
			bulk[3,1,3] = -1.
						
			bulk /= 2 
			
			r2 = zeros((4,2,2)) + 0.j
			r2[0,0,0] = 1.
			r2[1,1,1] = 1.
			
			r2[2,0,0] = -1.
			r2[3,1,1] = -1.
			
			r2 /= 2 ** (2. / 3.)
						
			r1 = zeros((2,2,1)) + 0.j
			r1[0,0,0] = 1.
			r1[1,0,0] = -1.
			r1 /= 2 ** (1. / 3.)
			
			self.matrices.append(l1.copy())
			self.matrices.append(l2.copy())
			for i in range(2, space_size - 2):
				self.matrices.append(bulk.copy())
			self.matrices.append(r2.copy())
			self.matrices.append(r1.copy())

		elif initial_state_type == "cluster":
			self.bondims = [4] * self._n
			self.bondims[0] = 2
			self.bondims[-1] = 1
			self.bondims[-2] = 2

			S_x = np.zeros((2,2))
			S_x[0][1] = 1
			S_x[1][0] = 1
			# S_y = np.array([[0, -1.j], [1.j, 0]])
			S_z = np.identity(2)
			S_z[1][1] = -1

			leftop = np.zeros((1, 2, 2, 2))
			centop = np.zeros((2, 2, 2, 2))
			rightop = np.zeros((2, 2, 2, 1))
			leftop[0, :, :, 0] = np.identity(2) * (2 ** (-1/3.))
			centop[0, :, :, 0] = np.identity(2) * (2 ** (-1/3.))
			rightop[0, :, :, 0] = np.identity(2) * (2 ** (-1/3.))
			leftop[0, :, :, 1] = S_z * (2 ** (-1/3.))
			centop[1, :, :, 1] = S_x * (2 ** (-1/3.))
			rightop[1, :, :, 0] = S_z * (2 ** (-1/3.))

			down = np.array([0, 1])

			bulk = np.tensordot(down, rightop, axes = [0, 1])[:, :, 0]
			bulk = np.tensordot(bulk, centop, axes = [1, 1])
			bulk = np.tensordot(bulk, leftop, axes = [2, 1])[:, :, :, 0, :, :]
			bulk = bulk.swapaxes(2, 3)
			dim = bulk.shape
			bulk = bulk.reshape(dim[0] * dim[1], dim[2], dim[3] * dim[4])

			leftone = np.tensordot(down, leftop, axes = [0, 1])

			lefttwo = np.tensordot(down, centop, axes = [0, 1])
			lefttwo = np.tensordot(lefttwo, leftop, axes = [1, 1])[:, :, 0, :, :]
			lefttwo = lefttwo.swapaxes(1, 2)
			dim = lefttwo.shape
			lefttwo = lefttwo.reshape(dim[0], dim[1], dim[2] * dim[3])

			righttwo = np.tensordot(down, rightop, axes = [0, 1])[:, :, 0]
			righttwo = np.tensordot(righttwo, centop, axes = [1, 1])
			dim = righttwo.shape
			righttwo = righttwo.reshape(dim[0] * dim[1], dim[2], dim[3])

			rightone = np.tensordot(down, rightop, axes = [0, 1])
			self.matrices = []
			self.matrices.append(leftone)
			self.matrices.append(lefttwo)
			for i in range(self._n - 4):
				self.matrices.append(bulk)
			self.matrices.append(righttwo)
			self.matrices.append(rightone)

		elif initial_state_type == 'AKLT':
			self._q = 3
			self.bondims = [2]*self._n
			self.bondims[-1] = 1
			bulk = np.zeros((2,3,2),dtype='d')
			bulk[1,0,0] = 2**(-0.5)
			bulk[0,1,0] = 0.5
			bulk[1,1,1] = -0.5
			bulk[0,2,1] = -2**(-0.5)
			
			self.matrices = []
			self.matrices.append(bulk[[1],:,:].copy()) # left spin1/2 edge state = \up
			for j in range(self._n - 2):
				self.matrices.append(bulk)
			self.matrices.append(bulk[:,:,[1]].copy()) # right spin1/2 edge state = \down

		
		self.id_uncanonical = None
		"""pointer to the A-tensor
		None: neither right-canon nor left-canon, need canonicalization
		in range(space_size-1): mixed-canonical form
			with self.matrix[id_uncanonical] being the only one not being cannonical
		"""
		self.merged_matrix = None
		self.merged_bond = None
		
	def __len__(self):
		return self._n

	def calcPsi(self, spinor_outcome):
		p = self._n - 1
		rvec = ones((1,),dtype=complex)
		if self.merged_matrix is None:
			while p >= 0:
				tmp = dot(spinor_outcome[p], self.matrices[p])
				rvec = dot(tmp, rvec)
				p -= 1
		else:
			mergedBdp1 = self.merged_bond + 1
			while p > mergedBdp1:
				tmp = dot(spinor_outcome[p], self.matrices[p])
				rvec = dot(tmp, rvec)
				p -= 1
			tmp = dot(spinor_outcome[p], self.merged_matrix)
			p -= 1
			tmp = dot(spinor_outcome[p], tmp)
			rvec = dot(tmp, rvec)
			p -= 1
			while p >= 0:
				tmp = dot(spinor_outcome[p], self.matrices[p])
				rvec = dot(tmp, rvec)
				p -= 1
		return rvec[0]

	def calcProb(self, spinor_outcome): 
		return np.abs(self.calcPsi(spinor_outcome)) ** 2.
	
	def mergeBond(self, bond):
		self.merged_bond = bond
		self.merged_matrix = tensordot(self.matrices[bond],self.matrices[bond + 1], ([2], [0]))
		
	def rebuildBond(self, going_right, keep_bondim=False):
		U, s, V = svd(reshape(self.merged_matrix, (self.bondims[(self.merged_bond - 1)] * self._q, self._q * self.bondims[(self.merged_bond + 1)])))
		if keep_bondim:
			bdm = min(self.bondims[self.merged_bond], s.size)
		else:
			if self.Dmin >= s.size:
				bdm = s.size
			else:
				bdm = max((s>s[0]*self.cutoff).sum(),self.Dmin)
		if self.Dmax is not None and self.Dmax < bdm:
			bdm = self.Dmax
		s = diag(s[:bdm])
		U = U[:, :bdm]
		V = V[:bdm, :]
		if going_right:
			V = dot(s, V)
			V/= norm(V)
			if self.id_uncanonical is not None: self.id_uncanonical = (self.merged_bond + 1)
		else:
			U = dot(U, s)
			U/= norm(U)
			if self.id_uncanonical is not None: self.id_uncanonical = self.merged_bond
		
		self.bondims[self.merged_bond] = bdm
		self.matrices[self.merged_bond] = reshape(U, (self.bondims[(self.merged_bond - 1) % self._n], self._q, bdm))
		self.matrices[(self.merged_bond + 1)] = reshape(V, (bdm, self._q, self.bondims[(self.merged_bond + 1)]))
		self.merged_bond = None
		self.merged_matrix=None
		return diag(s)
		
	def leftCano(self):
		if self.merged_bond is not None:
			self.rebuildBond(True)
		start = self.id_uncanonical if self.id_uncanonical is not None else 0
		for bond in range(start, self._n - 1):
			self.mergeBond(bond)
			self.rebuildBond(True, keep_bondim=True)
		self.id_uncanonical = self._n - 1

	def rightCano(self):
		if self.merged_bond is not None:
			self.rebuildBond(False)
		start = self.id_uncanonical-1 if self.id_uncanonical is not None else self._n-2
		for bond in range(start, -1, -1):
			self.mergeBond(bond)
			self.rebuildBond(False, keep_bondim=True)
		self.id_uncanonical = 0

	def genSample(self, spinor_setting):
		respin = zeros((self._n,self._q), complex)
		restate = 0
		assert self.id_uncanonical is not None
		p_unnorm = self.id_uncanonical
		assert p_unnorm<self._n and p_unnorm>=0
		# sampling order: p_unnorm, p_unnorm-1,...,0, p_unnorm+1,p_unnorm+2,...,N-1
		for stag, rang in enumerate([[p_unnorm],range(p_unnorm-1,-1,-1),range(p_unnorm+1,self._n)]):
			for pt in rang:
				mats_r = [dot(spinor_setting[pt,rm], self.matrices[pt]) for rm in range(self._q)]
				if stag == 1: # range(p_unnorm-1,-1,-1) accumulate from left
					mats_r = [dot(m, mat) for m in mats_r]
				elif stag ==2: # range(p_unnorm+1,self._n) accumulate from right
					mats_r = [dot(mat, m) for m in mats_r]
				norm2_mats_r = np.array([norm(m)**2 for m in mats_r])
				try:
					assert abs(norm2_mats_r.sum()-1)<1e-13
				except:
					raise ValueError('stag = %d, pt = %d, norm2_mat_r.sum() = %f'%(stag,pt,norm2_mats_r.sum()))
				rm = int(np.random.choice(np.arange(self._q),1,p=norm2_mats_r))
				if rm > 0:
					restate += rm*(self._q**pt)
				mat = mats_r[rm]/(norm2_mats_r[rm]**0.5)
				respin[pt,:] = spinor_setting[pt,rm,:]
		return restate, respin
		
	def calcFidelityTo(self, mats, persite=False):
		assert self._n == len(mats) and self._q == mats[0].shape[1]
		assert self.merged_bond is None
		if isinstance(mats, MPS):
			mats = mats.matrices
		p = self._n - 1
		res = dot(self.matrices[p][:,:,0],mats[p].conj())[:,:,0]
		while p>0:
			p -= 1
			res = dot(self.matrices[p],res)
			res = tensordot(res, mats[p].conj(),([1,2],[1,2]))
		if persite:
			return np.abs(res[0,0]) ** (1. / self._n)
		else:
			return np.abs(res[0,0])
	

class ProjMeasureSet:
	"""
	Outcomes of Projective Measurements
	You can either assign an MPS as generator or give the (spinor_settings, states) lists
	If an MPS is given, available measuring modes: uniform/2n+1/onlyZ/dynamic
	Attribute:
		noise: the noise level in the measurement. It's the probability that a random outcome is obtained
	"""
	def __init__(self, space_size, init_set_size=0, mode='uniform', mps_list=None, p_list=None, noise=0):
		self.__n = space_size
		self.__mps_ = mps_list
		self.__p_ = p_list / sum(p_list)
		self.__rho_rk = len(mps_list)
		assert self.__rho_rk == len(self.__p_)

		self.spinor_outcomes = []
		self.setMode(mode)
		self.noise = noise
		if init_set_size > 0:
			self.measureUpTo(init_set_size)

	def getN(self):
		return self.__n

	def designateMPSlist(self, mps_list, p_list):
		if self.states==[]:
			self.__mps_ = mps_list
			self.__p_ = p_list / sum(p_list)
			self.__rho_rk = len(mps_list)
			assert self.__rho_rk == len(self.__p_)
			for m in mps_list:
				m.leftCano()
		else:
			raise AttributeError('You cannot designate another MPS because there are measured data.')

	def _singlemeas(self):
		setting = self.__gen_setting()
		if self.noise > 0 and rand() < self.noise:
			binoutcome = randint(0,2,self.__n)
			return asarray([setting[j,binoutcome[j]] for j in range(self.__n)])
		else:
			jm = np.random.choice(np.arange(self.__rho_rk), p=self.__p_)
			return self.__mps_[jm].genSample(setting)[1]

	def setMode(self, mod):
		# if self.states == []:
		self.__mode = mod
		if mod=='uniform':
			self.__gen_setting = self.uniforMeas
		elif mod=='onlyZ':
			self.__gen_setting = self.zzMeas
		else:
			raise ValueError("Unknown measuring mode %s"%mod)
	
	def uniforMeas(self):
		setting = empty((self.__n, 2,2),dtype=complex)
		c = rand(self.__n)*2-1
		c1 = (0.5*(1+c))**0.5 #cos(theta/2)
		s1 = (0.5*(1-c))**0.5 #sin(theta/2)
		phi = rand(self.__n) * pi
		phas= exp(1.0j*phi)
		setting[:,0,0] = c1
		setting[:,0,1] = s1*phas
		setting[:,1,0] = -s1*phas.conj()
		setting[:,1,1] = c1
		return setting

	def zzMeas(self):
		setting = empty((self.__n, 2,2),dtype=int8)
		setting[:,0,0] = 1
		setting[:,1,1] = 1
		setting[:,1,0] = 0
		setting[:,0,1] = 0
		return setting
		
	def measureUpTo(self, size):
		for _ in range(size-len(self.spinor_outcomes)):
		# if size <= len(self.states): doing nothing
			spin_state = self._singlemeas()
			self.spinor_outcomes.append(spin_state)

	def calcL2dist_toMPSEnsemble(self, lps):
		"""
		Calculate the L2 distance between the mixed state that is generating samples and a given Locally Purified State
		"""
		innprod = 0
		for m, p in zip(self.__mps_, self.__p_):
			innprod += p*(lps.calcInnProdWith(m))
		tr_rho2 = (np.asarray(self.__p_)**2).sum()
		for jm, pj in enumerate(self.__p_[:-1]):
			mj = self.__mps_[jm]
			for km_jm, pk in enumerate(self.__p_[jm+1:]):
				mk = self.__mps_[jm+km_jm]
				tr_rho2 += 2*(pj*pk)*((mj.calcFidelityTo(mk))**2)
		return (tr_rho2 + lps.sqTr() - 2*innprod)**0.5

	def __getstate__(self):
		""" Return a dictionary of state values to be pickled """
		exclusion=['_ProjMeasureSet__mps_',"_ProjMeasureSet__p_"]
		mydict={}
		for key in self.__dict__.keys():
			if key not in exclusion:
				mydict[key] = self.__dict__[key]
		return mydict

	def save(self, name):
		
		mps_name = name.split('/')
		mps_name = '/'.join(mps_name[:-1])
		if len(mps_name) > 0:
			mps_name = mps_name+'/stdmps.pickle'
		else:
			mps_name = 'stdmps.pickle'
		if name[0]=='/':
			mps_name = '/'+mps_name
		# try:
		# 	fp = open(mps_name,'rb')
		# 	fp.close()
		# except FileNotFoundError:
		if not os.path.exists(mps_name):
			with open(mps_name,'wb') as fp:
				pickle.dump({"MPS_":self.__mps_,"p_":self.__p_}, fp)
			with open(mps_name+'.prob.bondim','w') as fp:
				for jm, m in enumerate(self.__mps_):
					print("#%d\t p=%g,"%(jm,self.__p_[jm]),m.bondims,file=fp)
		with open(name+".pickle","wb") as fsav:
			pickle.dump(self, fsav)

	def load(self, name):
		fload = open(name+'.pickle','rb')
		im = pickle.load(fload)
		for ky in im.__dict__.keys():
			self.__dict__[ky] = im.__dict__[ky]
		fload.close()
		# info = load(name+'.npz')
		# self.spinor_settings = list(info['spin'])
		# self.states = list(info['stat'])
		# assert self.__mode == str(info['mode'])

		mps_name = name.split('/')
		mps_name = '/'.join(mps_name[:-1])
		if len(mps_name) > 0:
			mps_name = mps_name+'/stdmps.pickle'
		else:
			mps_name = 'stdmps.pickle'
		if name[0]=='/':
			mps_name = '/'+mps_name
		with open(mps_name,'rb') as fmps:
			im = pickle.load(fmps)
			self._ProjMeasureSet__mps_ = im['MPS_']
			self._ProjMeasureSet__p_   = im['p_']

if __name__ == '__main__':
	pass
