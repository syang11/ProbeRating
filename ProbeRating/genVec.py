#!/usr/bin/python

"""
Created on Sat November 30 12:21:10 2019

@author: Shu Yang
"""
from Bio import SeqIO
import sys
#sys.path.append(**add your FastBioseq path here**)
import fastbioseq
import pandas as pd
import numpy as np


def gen_rnavec(mFile='./trained_models/exampleModel', faFile="example.fasta", vecFile='example.csv', modelTypeIndex=2):

	if modelTypeIndex==1:
		rv2 = fastbioseq.w2vbioseq.load_model(mFile) #biovec.models.w2vRNA.load_model(mFile)
	elif modelTypeIndex==2:
		rv2 = fastbioseq.ftbioseq.load_model(mFile) #biovec.models.ftRNA.load_model(mFile)
	elif modelTypeIndex==3:
		rv2 = fastbioseq.d2vbioseq.load_model(mFile) #biovec.models.load_protvec(mFile)
	else:
		raise RuntimeError, "modelTypeIndex (last argument to the script) should be [1 to 3]!"
	
	all_vectors=[]
	all_names=[]
	for seq_record in SeqIO.parse(faFile, "fasta"):
		tmp_seqs=seq_record.seq.split('*')
		vecsum=[]
		for tseq in tmp_seqs:
			if modelTypeIndex==3:
				vec=rv2.to_vecs(tseq)
			else:
				vec=sum(rv2.to_vecs(tseq))
			vecsum.append(vec)
		vector = fastbioseq.VecDF(sum(vecsum))
		
		all_vectors.append(vector)
		all_names.append(seq_record.id)

	allvecs=np.array([x.vec for x in all_vectors])
	allnam=np.array(all_names)
	df=pd.DataFrame(allvecs,allnam)
	df.to_csv(vecFile,header=0, index=0)	



if __name__ == '__main__':
	modelFile=sys.argv[1]
	fastaFile=sys.argv[2]
	vectorFile=sys.argv[3]
	modelTypeIndex=int(sys.argv[4])	# 1 for word2vec, 2 for fasttxt, 3 for doc2vec	

	gen_rnavec(modelFile, fastaFile, vectorFile, modelTypeIndex)