# -*- coding: utf-8 -*-

from numpy import dot, newaxis
import numpy as np
from numpy.random import rand, seed, randn, randint
from numpy.linalg import norm,eig,solve,svd
from copy import deepcopy
from os import mkdir,chdir
import shutil
import pickle
import os
import sys

def normalize(t):
	t /= norm(t)

class LPS:
	"""
	Locally Purified State
	Base LPS Class
	Parameter:
		initial_state_type:
			"W"/"dimer"/"cluster" and you will obtain the designated MPS state.
			"random": each entry drawn from uniform distribution U[0,1]
	Attributes:
		tensors:
			A list of rank 4 tensors. Indices: (auxilary, matrix-left, physical, matrix-right)
		id_uncanonical:
			-1: neither right-canon nor left-canon, need canonicalization
			in range(space_size-1): self.matrix[id_uncanonical] is the only one not being cannonical
			space_size-1: left-canonicalized
	"""
	def __init__(self, space_size, spinS=0.5, **kwarg):
		self._n = space_size
		self._q = int(2*spinS+1)
		self.cutoff = 0.01
		self.Dmin = 2
		if 'Dmin' in kwarg:
			self.Dmin = kwarg['Dmin']
		self.Dmax = None

		if "randomInitMaxD" in kwarg:
			self.bondims = randint(self.Dmin, kwarg['randomInitMaxD'],self._n)
		else:
			init_bond_dimension = 2
			self.bondims = [init_bond_dimension] * self._n ## bond[i] connect i i+1
		self.bondims[-1] = 1
		self.tensors = []
		self._q = kwarg['q']
		for i in range(space_size):
			self.tensors.append(randn(self._q, self.bondims[i-1], self._q, self.bondims[i]) + randn(self._q, self.bondims[i-1], kwarg['q'], self.bondims[i]) * 1.j)
			
		self.id_uncanonical = None
		"""pointer to the A-tensor
		None: neither right-canon nor left-canon, need canonicalization
		in range(space_size-1): mixed-canonical form
			with self.matrix[id_uncanonical] being the only one not being cannonical
		"""
		self.merged_tensor = None
		self.merged_bond = None
		
	def __len__(self):
		return self._n

	def calcProb(self, spinor_outcome):
		p = self._n - 1 # Point at the rightmost tensor
		rtensor = ones((1,1),dtype=complex)
		if self.merged_tensor is None:
			while p >= 0:
				tmp = dot(spinor_outcome[p], self.tensors[p]) # contract the spinor index with the physical bond
				# now tmp has 3 indices: (auxilary, matrix-left, matrix-right)
				rtensor = dot(tmp.conj(), rtensor)
				# now rtensor temporarily has 3 indices: (auxilary, matrix-left, old-rtensor-lower-leg)
				rtensor = np.tensordot(rtensor, tmp, axes=([0,2],[0,2]))
				p -= 1
		else:
			mergedBdp1 = self.merged_bond + 1
			while p > mergedBdp1:
				tmp = dot(spinor_outcome[p], self.tensors[p]) # contract the spinor index with the physical bond
				# now tmp has 3 indices: (auxilary, matrix-left, matrix-right)
				rtensor = dot(tmp.conj(), rtensor)
				# now rtensor temporarily has 3 indices: (auxilary, matrix-left, old-rtensor-lower-leg)
				rtensor = np.tensordot(rtensor, tmp, axes=([0,2],[0,2]))
				p -= 1
			p = self.merged_bond
			tmp = np.tensordot(spinor_outcome[p], dot(spinor_outcome[mergedBdp1], self.merged_tensor), axes=([0],[2]))
			# now tmp has 4 indices: (auxilary, matrix-left, auxilary, matrix-right)
			tmp = np.tensordot(tmp.conj(), tmp, axes=([0,2],[0,2]))
			# now tmp has 4 indices: (conj-matrix-left, conj-matrix-right, matrix-left, matrix-right)
			rtensor = np.tensordot(tmp, rtensor, axes=([1,3],[0,1]))
			p -= 1

			while p >= 0:
				tmp = dot(spinor_outcome[p], self.tensors[p]) # contract the spinor index with the physical bond
				# now tmp has 3 indices: (auxilary, matrix-left, matrix-right)
				rtensor = dot(tmp.conj(), rtensor)
				# now rtensor temporarily has 3 indices: (auxilary, matrix-left, old-rtensor-lower-leg)
				rtensor = np.tensordot(rtensor, tmp, axes=([0,2],[0,2]))
				p -= 1
		assert rtensor.shape == (1,1)
		return rtensor[0,0]
	
	def mergeBond(self, bond):
		self.merged_bond = bond
		self.merged_tensor = np.tensordot(self.tensors[bond], self.tensors[bond+1], ([3],[1]))
		# the merged_tensor has 6 indices: (auxilary, mat-left, physical, auxilary, physical, mat-right)
		
	def rebuildBond(self, going_right, keep_bondim=False):
		# try:
		# 	mreshape = reshape(self.merged_tensor, (self.bondims[(self.merged_bond - 1)] * self._q, self._q * self.bondims[(self.merged_bond + 1)]))
		# except:
		# 	raise ValueError('self.merged_tensor.shape =',self.merged_tensor.shape, "self._q =", self._q, (self.bondims[(self.merged_bond - 1)] * self._q, self._q * self.bondims[(self.merged_bond + 1)]))
		U, s, V = svd(reshape(self.merged_tensor, 
			(self.bondims[(self.merged_bond-1)] * (self._q**2), (self._q**2) * self.bondims[(self.merged_bond + 1)])))
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
			normalize(V)
			if self.id_uncanonical is not None: self.id_uncanonical = (self.merged_bond + 1)
		else:
			U = dot(U, s)
			normalize(U)
			if self.id_uncanonical is not None: self.id_uncanonical = self.merged_bond
		
		self.bondims[self.merged_bond] = bdm
		self.tensors[self.merged_bond] = reshape(U, (self._q, self.bondims[(self.merged_bond - 1) % self._n], self._q, bdm))
		self.tensors[(self.merged_bond + 1)] = reshape(V, (self._q, bdm, self._q, self.bondims[(self.merged_bond + 1)]))
		self.merged_bond   = None
		self.merged_tensor = None
		return np.diag(s)
		
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
				tens_r = []
				for rm in range(self._q):
					m = dot(spinor_setting[pt,rm], self.tensors[pt])
					tens_r.append(np.tensordot(m.conj(), m, axes=([0],[0])))
				if stag == 1: # range(p_unnorm-1,-1,-1) accumulate from left
					tens_r = [np.tensordot(t, tens, axes=([1,3],[0,2])) for t in tens_r]
				elif stag == 2: # range(p_unnorm+1,self._n) accumulate from right
					tens_r = [np.tensordot(tens, t, axes=([1,3],[0,2])) for t in tens_r]
				bdm_prod = tens_r[0].shape
				bdm_prod = bdm_prod[0] * bdm_prod[1]
				norm2_r = np.array([np.trace(t.reshape(bdm_prod, bdm_prod)) for t in tens_r])
				try:
					assert abs(norm2_mats_r.sum()-1)<1e-13
				except:
					raise ValueError('stag = %d, pt = %d, norm2_mat_r.sum() = %f'%(stag,pt,norm2_mats_r.sum()))
				rm = int(np.random.choice(np.arange(self._q), 1, p=norm2_r))
				if rm > 0:
					restate += rm*(self._q**pt)
				tens = tens_r[rm] / (norm2_r[rm])
				respin[pt,:] = spinor_setting[pt,rm,:]
		return restate, respin

	def calcInnProdWith(ps):
		if isinstance(ps, MPS):
			tensors = [m[newaxis,:,:,:] for m in ps.matrices]
		elif isinstance(ps, LPS):
			tensors = ps.tensors
		else:
			tensors = None
		ltens = np.ones((1,1), dtype=complex)
		for p in range(self._n):
			# physical bond contraction
			tmp = np.tensordot(self.tensor[p].conj(), tensors[p], axes=([2],[2])).transpose([0,3,1,4,2,5])
			s = tmp.shape
			tmp.shape = (s[0]*s[1], s[2]*s[3], s[4]*s[5])
			# (aux, mat-left, mat-right)
			ltens = np.dot(ltens, tmp)
			# (conj-mat-right, aux, mat-right)
			ltens = np.tensordot(ltens, tmp.conj(), axes=([0,1],[1,0]))
		return np.trace(ltens)

	def calcL2dist_toLPS(lqs):
		dist_sq = self.sqTr() + lqs.sqTr() - 2*self.calcInnProdwith(lqs)
		return dist_sq**0.5

	def sqTr(self):
		return self.calcInnProdwith(self)

	
# up-to-date
class MLETomoTrainer(LPS):
	"""
	Tomograpy Trainer:
	Attribute:
		dat:	the dataset containing the outcome batch
		dat_head, dat_rear:		indices defining where the cumulants concern -- dat[dat_head:dat_rear]
		_cumulantL, _cumulantR:	1) _cumulantL[0] == ones((n_sample, 1))
								2) _cumulantR[0] == ones((1, n_sample))
								3) if j>0: _cumulantL[j] = A(0)...A(j-1)
										   _cumulantR[j] = A(N-j)...A(N-1)

	When saving, dat won't be saved along with a TomoTrainer instance
	so when loading, dat need extra attaching, using attach_dat
	"""
	def __init__(self, dataset, batch_size=40, initV=80, add_mode=True, **kwarg):
		LPS.__init__(self, dataset.getN(), **kwarg)
		self.leftCano()
		self.loss = []
		self.succ_fid = []
		self.real_fid = []
		self.train_history = []
		self.ibatch = -1
		self.add_mode = add_mode
		self.batch_size = batch_size
		self.dat_head = 0
		self.dat_rear = initV
		self.attach_dat(dataset)

		self.grad_mode = 'plain' # plain/gnorm/RMSProp/Adam
		self.learning_rate = 0.1
		self.descent_steps = 2
		self.penalty_rate = None

		# self.loss.append(self.calcLoss(dataset))
	def attach_dat(self, dataset):
		self.dat = dataset
		dataset.measureUpTo(self.dat_rear)

	def _showProb(self, istate):
		"""Evaluate with cumulant"""
		spin = self.dat.spinor_outcomes[istate]
		istate -= self.dat_head
		lvec = self._cumulantL[-1][istate].ravel()
		rtensor = self._cumulantR[-1][istate]
		k = self.merged_bond
		if k is not None:
			tmp = np.tensordot(spin[k], np.dot(spin[k+1], self.merged_tensor), axes=([0],[2]))
			tmp = np.tensordot(tmp.conj(), tmp, axes=([0,2],[0,2])) # auxilary bonds contraction
			tmp = tmp.swapaxes(1,2)
			s = tmp.shape
			tmp.shape = (s[0]*s[1],s[2]*s[3])
			tmp = np.dot(tmp, rtensor.ravel())
		else:
			tmp = rtensor
			for k in [len(self._cumulantL), len(self._cumulantL)-1]:
				tmp1 = np.dot(spin[k], self.tensors[k])
				tmp = np.tensordot(np.dot(tmp1.conj(), tmp), tmp1, axes=([0,2],[0,2]))
			tmp = tmp.ravel()
		return lvec.dot(tmp)

	def _showLoss(self):
		"""Evaluate with cumulant"""
		res = -mean([log(self._showProb(i)) for i in range(self.dat_head, self.dat_rear)])
		print('Loss =',res)
		return res

	def calcLoss(self, dataset=None, head=0, rear=-1):
		"""
		Evaluate NLL on dataset
		if dataset is None:
			dataset = self.dat
			and it will call self._showPsi, using cumulant
		else: 
			given dataset
		"""
		if dataset is None:
			dataset = self.dat
		if rear == -1:
			rear = len(dataset.states)
		if dataset is not None:
			res = -mean([log(self.calcProb(dataset.spinor_outcomes[i])) for i in range(head, rear)])
		else:
			res = -sum([log(np.abs(self._showProb(i))) for i in range(max(self.dat_head,head), min(rear,self.dat_rear))])
			for i in range(head, max(self.dat_head,head)):
				res -= log(self.calcProb(dataset.spinor_outcomes[i]))
			for i in range(min(rear,self.dat_rear), rear):
				res -= log(self.calcProb(dataset.spinor_outcomes[i]))
			res /= (rear-head)
		print('Loss =',res)
		return res

	def _initCumulant(self):
		"""
		Initialize self.cumulants for a batch.
		During the training process, it will be kept unchanged that:
		1) _cumulantL[0] == ones((n_sample, 1))
		2) _cumulantR[0] == ones((1, n_sample))
		3) if j>0: _cumulantL[j] = A(0)...A(j-1)
				   _cumulantR[j] = A(N-j)...A(N-1)
		"""
		if self.id_uncanonical == self._n-1:
			self.leftCano()
		self._cumulantL = [np.ones((self.dat_rear-self.dat_head, 1,1), dtype=complex)]
		for n in range(0, self._n-2):
			tmp = []
			for i in range(self.dat_head, self.dat_rear):
				mid1 = np.dot(self.dat.spinor_outcomes[i][n], self.tensors[n])
				mid = np.tensordot(np.tensordot(mid1.conj(), self._cumulantL[-1][i-self.dat_head], axes=([1],[0])), mid1, axes=([0,2],[0,1]))
				tmp.append(mid)
			self._cumulantL.append(np.asarray(tmp))
		self._cumulantR = [np.ones((self.dat_rear-self.dat_head, 1,1), dtype=complex)]

	def _updateCumulant(self, going_right):
		k = len(self._cumulantL)-1
		if going_right:
			tmp = []
			for i in range(self.dat_head, self.dat_rear):
				mid1 = np.dot(self.dat.spinor_outcomes[i][k], self.tensors[k])
				mid = np.tensordot(np.tensordot(mid1.conj(), self._cumulantL[-1][i-self.dat_head], axes=([1],[0])), mid1, axes=([0,2],[0,1]))
				tmp.append(mid)
			self._cumulantL.append(np.asarray(tmp))
			self._cumulantR.pop()
		else:
			k += 1
			tmp = []
			for i in range(self.dat_head, self.dat_rear):
				mid1 = np.dot(self.dat.spinor_outcomes[i][k], self.tensors[k])
				mid = np.tensordot(np.dot(mid1.conj(), self._cumulantR[-1][i-self.dat_head]), mid1, ([0,2],[0,2]))
				tmp.append(mid)
			self._cumulantR.append(asarray(tmp))
			self._cumulantL.pop()

	def _neGrad(self):
		"""negative gradient"""
		k = self.merged_bond
		grad = zeros((self.bondims[(k-1) % self._n], 2, 2, self.bondims[(k+1) % self._n]), dtype=complex)
		cumuL = self._cumulantL[-1]
		cumuR = self._cumulantR[-1]
		for istate in range(self.dat_head, self.dat_rear):
			spin = self.dat.spinor_outcomes[istate]
			pprime = np.tensordot(spin[k], np.dot(spin[k+1], self.merged_tensor), axes=([0],[2]))
			pprime = np.tensordot(np.tensordot(cumuL[istate-self.dat_head], pprime, axes=([1],[1])), 
									cumuR[istate-self.dat_head], axes=([3],[1]))
			# indices: (conj-Ak-left, Ak-aux, Ak+1-aux, conj-Ak+1-right)
			pprime = (pprime.swapaxes(0,1))[:,:,newaxis,:,newaxis,:]
			# indices: (Ak-aux, conj-Ak-left, conj-Ak-phys, Ak+1-aux, conj-Ak+1-phys, conj-Ak+1-right)
			pprime = spin[k].conj()[newaxis,newaxis,:,newaxis,newaxis,newaxis] * pprime * spin[k+1].conj()[newaxis,newaxis,newaxis,newaxis,:,newaxis]
			probi  = pprime.ravel().dot(self.merged_tensor.conj().ravel())
			grad += pprime/probi
		grad /= (self.dat_rear - self.dat_head)
		grad -= conj(self.merged_tensor) 
		
		# if self.penalty_rate is not None and self.penalty_rate != 0.:
		# 	S2_penalty = np.tensordot(conj(self.merged_tensor), self.merged_tensor, ([2,3], [2,3]))
		# 	S2_penalty = np.tensordot(S2_penalty, conj(self.merged_tensor), ([2,3], [0,1]))
		# 	expS2 = np.tensordot(S2_penalty, self.merged_tensor, ([0,1,2,3], [0,1,2,3]))
		# 	conjgrad += S2_penalty / expS2 * self.penalty_rate
		
		return grad

	def _multiSteps(self):
		if self.grad_mode == 'plain':
			for j in range(self.descent_steps):
				ngrad = self._neGrad()
				self.merged_tensor += ngrad * self.learning_rate
				if j < self.descent_steps-1:
					normalize(self.merged_tensor)
		elif self.grad_mode == 'gnorm':
			for j in range(self.descent_steps):
				ngrad = self._neGrad()
				gnorm = norm(ngrad)/((ngrad.size)**0.25)
				if gnorm < 1:
					ngrad /= gnorm
				self.merged_tensor += ngrad * self.learning_rate
				if j < self.descent_steps-1:
					normalize(self.merged_tensor)
		elif self.grad_mode == 'RMSProp' or self.grad_mode=='RMSprop':
			rho = 0.9
			delta = 1e-15
			r = np.zeros(self.merged_tensor.shape)
			for j in range(self.descent_steps):
				ngrad = self._neGrad()
				r = (1-rho)*ngrad*ngrad + rho*r
				self.merged_tensor += (ngrad/np.sqrt(delta+r))*self.learning_rate
				if j < self.descent_steps-1:
					normalize(self.merged_tensor)
		elif self.grad_mode == 'RMSProp-momentum':
			rho = 0.9
			alpha = 0.8
			delta = 1e-15
			r = np.zeros(self.merged_tensor.shape)
			v = np.zeros(self.merged_tensor.shape)
			for j in range(self.descent_steps):
				if j > 0:
					self.merged_tensor += alpha*v
					normalize(self.merged_tensor)
				ngrad = self._neGrad()
				r = (1-rho)*ngrad*ngrad + rho*r
				v = alpha*v + ngrad/np.sqrt(r)*self.learning_rate
				self.merged_tensor += v
		elif self.grad_mode == 'Adam' or self.grad_mode == 'adam':
			rho1 = 0.9
			rho2 = 0.99
			delta = 1e-15
			s = np.zeros(self.merged_tensor.shape)
			r = np.zeros(self.merged_tensor.shape)
			for j in range(1,self.descent_steps+1):
				ngrad = self._neGrad()
				s = rho1*s - (1-rho1)*ngrad
				r = rho2*r + (1-rho2)*ngrad*ngrad
				self.merged_tensor -= self.learning_rate * s/(1-rho1**j)/(delta+np.sqrt(r/(1-rho2**j)))
				if j < self.descent_steps:
					normalize(self.merged_tensor)

	def train(self, loops):
		tmp_rear = self.batch_size + self.dat_rear
		self.dat.measureUpTo(tmp_rear)
		if not self.add_mode:
			self.dat_head = self.dat_rear
		self.dat_rear = tmp_rear
		late_lps = deepcopy(self)
		self._initCumulant()
		for lp in range(loops):
			for b in range(self._n-2, 0, -1):
				self.mergeBond(b)
				self._multiSteps()
				self.rebuildBond(False)
				self._updateCumulant(False)
			# self.calcLoss()
			for b in range(0, self._n-2):
				self.mergeBond(b)
				self._multiSteps()
				self.rebuildBond(True)
				self._updateCumulant(True)
			# self.calcLoss()
		self.ibatch += 1
		self.train_history.append((self.cutoff, self.descent_steps, self.learning_rate, self.penalty_rate, self.dat_head, self.dat_rear, loops))	
		
		self.dist.append(self.calcL2dist_toLPS(lat_lps))
		self.real_dist.append(self.dat.calcL2dist_toMPSEnsemble(self))

	def save(self, stamp):
		try:
			mkdir('./'+stamp+'/')
		except:
			shutil.rmtree(stamp)
			mkdir('./'+stamp+'/')
		chdir('./'+stamp+'/')
		# fp = open('MPS.log', 'w')
		# fp.write("Present State of MPS:\n")
		# fp.write("space_size=%d\t,cutoff=%.10f\t,step_len=%f\n"% (self._n, self.cutoff,self.learning_rate)) 
		# fp.write("bond dimension:"+str(self.bondims))
		# fp.write("\tloss=%1.6e\n"%self.loss[-1])
		save('Bondim.npy',self.bondims)
		save('Mats.npy',self.tensors)
		save('ibatch.npy',self.ibatch)

		# print('Saved')
		# fp.write("cutoff\tn_descent\tstep_length\tpenalty\t(dat_h, dat_r)\tn_loop\n")
		# for history in self.train_history:
		# 	fp.write("%1.2e\t%d\t%1.2e\t%1.2e\t(%d,%d)\t%d\n"%tuple(history))
		# fp.close()
		chdir('..')
		# save('Loss.npy',self.loss)
		savez('Fidelity.npz', succ=self.succ_fid, real=self.real_fid)
		with open('TrainHistory.pickle', 'wb') as thp:
			pickle.dump(self.train_history, thp)

		# self.dat.save('dataset.npz')
	
	def load(self, srch_pwd=None):
		if srch_pwd is not None:
			oripwd = os.getcwd()
			os.chdir(srch_pwd)
		self.bondims = load('Bondim.npy').tolist()
		self.__n = len(self.bondims)
		try:
			self.Loss = load('Loss.npy').tolist()
		except FileNotFoundError:
			self.Loss = []

		try:
			self.ibatch = np.load('ibatch.npy')
			# if self.ibatch ==
		except FileNotFoundError:
			if srch_pwd is None:
				raise ValueError("ibatch can't be found")
			else:
				self.ibatch = readLfromdir(srch_pwd)
		try:
			with open('../TrainHistory.pickle', 'rb') as thp:
				self.train_history = pickle.load(thp)[:self.ibatch+1]
		except:
			self.train_history = load('TrainHistory.npy').tolist()
			self.train_history = [(t[0],int(t[1]),t[2],None if abs(t[3])<1e-10 else t[3],\
							   int(t[4]), int(t[5]), int(t[6])) for t in self.train_history]
		try:
			self.cutoff,self.descent_steps,self.learning_rate,self.penalty_rate,\
				self.dat_head, self.dat_rear, _lp = self.train_history[-1]
		except:
			raise IndexError("len(self.train_history)=%d, ibatch=%d"%(len(self.train_history),self.ibatch))
		# self.descent_steps = int(self.descent_steps)
		# if abs(self.penalty_rate) < 1e-10:
		# 	self.penalty_rate = None
		# self.dat_head = int(self.dat_head)
		# self.dat_rear = int(self.dat_rear)

		try:
			fids = np.load('../Fidelity.npz')
		except:
			fids = np.load('Fidelity.npz')
		self.succ_fid = fids['succ'].tolist()[:self.ibatch+1]
		self.real_fid = fids['real'].tolist()[:self.ibatch+1]
		self.tensors = load('Mats.npy').tolist()
		
		self.merged_bond = None
		self.merged_tensor=None
		if srch_pwd is not None:
			os.chdir(oripwd)

def readLfromdir(srch_pwd):
	if srch_pwd[-1]=='/' or srch_pwd[-1]=='_':
		srch_pwd = srch_pwd[:-1]
	k = -1
	while k > -len(srch_pwd):
		k -= 1
		if srch_pwd[k] == 'L':
			k += 1
			break
	k1 = k
	while k1 <= -1:
		if srch_pwd[k1] == 'R':
			break
		k1 += 1
	if k1==0:
		return int(srch_pwd[k:])
	else:
		return int(srch_pwd[k:k1])

def preparation(typ, nn, tot):
	sm = MPS(nn,typ)
	ds = ProjMeasureSet(nn, tot, mps=sm)
	return ds

if __name__ == '__main__':
	pass
	# typ = sys.argv[1]
	# space_size = int(sys.argv[2])
	# sm = MPS(space_size, typ)
	# comm = MPI.COMM_WORLD
	# rk = comm.Get_rank()
	# seed(rk)
	
	# mxBatch = 200
	# batch_size = 40
	# ds = preparation(typ, space_size, mxBatch*batch_size)

	# measout = measout_dir+'/%s/%d/'%(typ, space_size)
	# try:
	# 	mkdir(measout)
	# except FileExistsError:
	# 	pass
	# ds.save(measout+"/R%dSet"%(rk))

	# fload = open("N%dL%dR%dSet"%(space_size,mxBatch-1,rk)+'.pickle','rb')
	# ds1 = pickle.load(fload)
	# fload.close()
	# # sm1 = MPS(20,'W')
	# # ds1 = ProjMeasureSet(20,0,mps=sm1)
	# # ds1.load("N%dL%dR%dSet"%(space_size,mxBatch-1,rk))
	# print('#%d'%rk, ds1.getN())


	# dat = ProjMeasureSet(space_size, init_set_size=80, mode='uniform', mps = sm)

	# m = TomoTrainer(dat)
	# m.add_mode = True
	# m.grad_mode = argv[1]
	# m.batch_size = 40
	# m.cutoff = 0.8
	# m.descent_steps = 10
	# m.learning_rate = float(argv[2])
	# m.penalty_rate = None
	
	# mxBatch = 160
	# for b in range(mxBatch):
	# 	m.train(nloop)
	# 	if b%10==9:
	# 		stmp = 'L%d'%b
	# 		m.save(stmp)
