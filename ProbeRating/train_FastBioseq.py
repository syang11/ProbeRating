"""
Created on Sat November 30 17:30:01 2019

@author: Shu Yang
"""

import sys
#sys.path.append(**add your FastBioseq path here**)
import fastbioseq
from memory_profiler import profile
from guppy import hpy

@profile
def train_ftbioseq(inFile='example.fasta', outFile='./trained_models/exampleModel', vecDim=50, k=3, minCount=2):
	'''
	FastBioseq model trained with FastText
	'''
	pv = fastbioseq.ftbioseq(inFile, n=k, size=vecDim, min_count=minCount, min_n=2, max_n=k) 
	pv.save(outFile)
	h = hpy()
	print h.heap()
	return None #for memory profile purpose



@profile
def train_d2vbioseq(inFile='example.fasta', outFile='exampleModel', vecDim=50, k=3, minCount=1):
	'''
	FastBioseq model trained with doc2vec
	'''
	pv = fastbioseq.d2vbioseq(inFile, n=k, size=vecDim, min_count=minCount) 
	pv.save(outFile)
	h = hpy()
	print h.heap()
	return None #for memory profile purpose



@profile
def train_w2vbioseq(inFile='example.fasta', outFile='exampleModel', vecDim=50, k=3, minCount=1):
	'''
	FastBioseq model trained with word2vec
	'''
	pv = fastbioseq.w2vbioseq(inFile, n=k, size=vecDim, min_count=minCount) 
	pv.save(outFile)
	h = hpy()
	print h.heap()
	return None #for memory profile purpose



if __name__ == '__main__':
	fastaFile=sys.argv[1]
	modelFile=sys.argv[2]
	vecDimension=int(sys.argv[3])
	howManyMer=int(sys.argv[4])	
	minKmerCount=int(sys.argv[5])	
	train_ftbioseq(fastaFile, modelFile, vecDimension, howManyMer, minKmerCount)
	#train_d2vbioseq(fastaFile, modelFile, vecDimension, howManyMer, minKmerCount)
	#train_w2vbioseq(fastaFile, modelFile, vecDimension, howManyMer, minKmerCount)