"""
Created on Sat November 30 17:30:01 2019

@author: Shu Yang
"""

import sys
import biovec
from memory_profiler import profile
from guppy import hpy

@profile
def train_protvec(inFile='example.fasta', outFile='exampleModel', vecDim=100, k=3, minCount=2):
	'''(string, string, int) -> None
	'''
	pv = biovec.ProtVec(inFile, n=k, size=vecDim, min_count=1, min_n=1, max_n=2) #pv = biovec.ProtVec(inFile, size=vecDim)	#size controls the embedding dimension
	pv.save(outFile)
	h = hpy()
	print h.heap()
	return None #for memory profile purpose

if __name__ == '__main__':
	fastaFile=sys.argv[1]
	modelFile=sys.argv[2]
	vecDimension=int(sys.argv[3])
	howManyMer=int(sys.argv[4])	
	minWordCount=int(sys.argv[5])	
	train_protvec(fastaFile, modelFile, vecDimension, howManyMer, minWordCount)
