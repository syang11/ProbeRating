"""
Created on Sat November 30 10:57:52 2019

@author: Shu Yang
"""
import os
from Bio import SeqIO
import sys
from gensim.models import fasttext
from gensim.models import doc2vec
from gensim.models import word2vec
from gensim.utils import simple_preprocess


def split_ngrams(seq, n):
    """
    Work for protein or nucleic acid sequences.
    n==3: 'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    n==4: 'AGAMQSASM' => [['AGAM', 'QSAS'], ['GAMQ', 'SASM'], ['AMQS'], ['MQSA']]
    n==5: ...
    """
    frames=[]
    if n==3:
        a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
        frames=[a,b,c]
    elif n==4:
        a, b, c, d = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n), zip(*[iter(seq[3:])]*n)
        frames=[a,b,c,d]
    elif n==5:
        a, b, c, d, e = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n), zip(*[iter(seq[3:])]*n), zip(*[iter(seq[4:])]*n)
        frames=[a,b,c,d,e]
    elif n==6:
        a, b, c, d, e, f = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n), zip(*[iter(seq[3:])]*n), zip(*[iter(seq[4:])]*n), zip(*[iter(seq[5:])]*n)
        frames=[a,b,c,d,e,f]
    else:
        raise Exception("k of k-mer should be [3-6]!")

    str_ngrams = []
    for ngrams in frames:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams




def generate_corpusfile(fname, n, out):
    '''
    Args:
        fname: corpus file name
        n: n of n-gram, i.e. length of k-mer
        out: output corpus file path
    Description:
        load corpus file to generate input corpus.
    '''
    f = open(out, "w")
    for r in SeqIO.parse(fname, "fasta"):
        ngram_patterns = split_ngrams(r.seq, n)
        for ngram_pattern in ngram_patterns:
            f.write(" ".join(ngram_pattern) + "\n")
        sys.stdout.write(".")

    f.close()



def generate_corpusfile_overlappedKmers(fname, n, out):
    """
    For d2vbioseq's doc2vec version.

    Differing from generate_corpusfile() which is non-overlapping for each frame (i.e. if n==3, for each sequence, output 3 non-overlapped frames), n==3: 'AGAMQSASM' => "AGA MQS ASM\nGAM QSA\nAMQ SAS", here do the overlapped version:
    n==3: 'AGAMQSASM' => 'AGA GAM AMQ MQS QSA SAS ASM'
    """
    f = open(out, "w")
    for r in SeqIO.parse(fname, "fasta"):
        ngram_patterns = split_ngrams(r.seq, n)  
        doc_seq=''      
        #extract overlapped ngrams:        
        for tmp in map(None, *ngram_patterns):
            rmNone_tmp=map(lambda x: "" if x is None else x, tmp)
            doc_seq+=' '.join(rmNone_tmp) + ' '
        doc_seq.strip()
        f.write(doc_seq + "\n")
        sys.stdout.write(".")

    f.close()



def read_corpus(fname, tokens_only=False):
    '''
    Not used
    For d2vbioseq's doc2vec, pre-process each file and return a list of words
    '''
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])




class embedbioseq(object):
    def __init__(self, fname=None, corpus=None, n=3, size=50):
        """
        Either fname or corpus is required.
        fname: fasta file for corpus
        corpus: corpus object implemented by gensim
        n: n of n-gram. I use n=3 for protein, n=5 for nucleic acid.
        size: embedded vector dimension
        """
        self.n = n
        self.size = size
        self.fname = fname

        if corpus is None and fname is None:
            raise Exception("Either fname or corpus is needed!")


    def to_vecs(self, seq):
        """
        convert sequence to n-length vectors
        e.g. n=3 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ]
        """
        pass





class ftbioseq(embedbioseq, fasttext.FastText):
    """
    FastBioseq model trained with gensim FastText
    """
    def __init__(self, fname=None, corpus=None, n=3, size=50, out="ftBioseqCorpus-"+str(os.getpid())+".txt",  sg=0, window=25, min_count=2, min_n=2, max_n=3):
        """
        out: corpus output file path        
        sg: trianing algorithm, 0 for CBOW, 1 for skip-gram
        window: context window half width
        min_count: ignores all ngrams with total frequency lower than this
        min_n: minimum length of sub-word inside a n-gram/k-mer
        max_n: maximum length of sub-word inside a n-gram/k-mer

        More parameter options can be found on Gensim webpage
        """
        embedbioseq.__init__(self, fname, corpus, n, size)

        if fname is not None:
            print 'Generate Corpus file from fasta file...'
            generate_corpusfile(fname, n, out)
            corpus = word2vec.Text8Corpus(out)

        fasttext.FastText.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count, min_n=min_n, max_n=max_n)

        #syang: clean up the intermediate file
        if os.path.exists(out):
            os.remove(out)


    def to_vecs(self, seq):
        ngram_patterns = split_ngrams(seq, self.n)

        seqvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self[ngram])
                except:
                    # this exception should never happen though; keep to check correctness
                    raise Exception("OOV! FastText-FastBioseq model has never trained this k-mer: " + ngram)
            seqvecs.append(sum(ngram_vecs))
        return seqvecs


    @staticmethod
    def load_model(model_fname):
        return fasttext.FastText.load(model_fname)





class d2vbioseq(embedbioseq, doc2vec.Doc2Vec):
    """
    FastBioseq model trained with gensim doc2vec
    """
    def __init__(self, fname=None, corpus=None, n=3, size=50, out="d2vBioseqCorpus-"+str(os.getpid())+".txt",  dm=0, window=25, min_count=2):
        """
        out: corpus output file path        
        dm: trianing algorithm, 0 for PV-DBOW, 1 for PV-DM
        window: context window half width
        min_count: ignores all ngrams with total frequency lower than this

        More parameter options can be found on Gensim webpage
        """
        embedbioseq.__init__(self, fname, corpus, n, size)

        if fname is not None:
            print 'Generate Corpus file from fasta file...'
            generate_corpusfile_overlappedKmers(corpus_fname, n, out)
            corpus=[doc2vec.TaggedDocument(simple_preprocess(line), [i]) for i, line in enumerate(open(out))]

        doc2vec.Doc2Vec.__init__(self, corpus, vector_size=size, dm=dm, window=window, min_count=min_count)

        #syang: clean up the intermediate file
        if os.path.exists(out):
            os.remove(out)


    def to_vecs(self, seq):
        """
        Differing from ftbioseq and w2vbioseq's to_vecs() which converts sequence to n-length vectors
        e.g. 'AGAMQSASM' => [ array([  ... * 100 ], array([  ... * 100 ], array([  ... * 100 ] ], here do the overlapped version and returns only one vector for the sequence document

        refer to gensim's doc2vec document: https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
        """
        ngram_patterns = split_ngrams(seq, self.n)
        doc_seq=''      
        #extract overlapped ngrams:        
        for tmp in map(None, *ngram_patterns):
            rmNone_tmp=map(lambda x: "" if x is None else x, tmp)
            doc_seq+=' '.join(rmNone_tmp) + ' '
        doc_seq.strip()

        return self.infer_vector(doc_seq.split())


    @staticmethod
    def load_model(model_fname):
        return doc2vec.Doc2Vec.load(model_fname)





class w2vbioseq(embedbioseq, word2vec.Word2Vec):
    """
    FastBioseq model trained with gensim word2vec
    """
    def __init__(self, fname=None, corpus=None, n=3, size=50, out="w2vBioseqCorpus-"+str(os.getpid())+".txt",  sg=0, window=25, min_count=2):
        """
        out: corpus output file path        
        sg: trianing algorithm, 0 for CBOW, 1 for skip-gram
        window: context window half width
        min_count: ignores all ngrams with total frequency lower than this

        More parameter options can be found on Gensim webpage
        """
        embedbioseq.__init__(self, fname, corpus, n, size)

        if fname is not None:
            print 'Generate Corpus file from fasta file...'
            generate_corpusfile(fname, n, out)
            corpus = word2vec.Text8Corpus(out)

        word2vec.Word2Vec.__init__(self, corpus, size=size, sg=sg, window=window, min_count=min_count)

        if os.path.exists(out):
            os.remove(out)


    def to_vecs(self, seq):        
        ngram_patterns = split_ngrams(seq, self.n)

        seqvecs = []
        for ngrams in ngram_patterns:
            ngram_vecs = []
            for ngram in ngrams:
                try:
                    ngram_vecs.append(self[ngram])
                except:
                    raise Exception("OOV! Word2vec-FastBioseq model has never trained this k-mer: " + ngram)
            seqvecs.append(sum(ngram_vecs))
        return seqvecs


    @staticmethod
    def load_model(model_fname):
        return word2vec.Word2Vec.load(model_fname)



class VecDF:
    """
    Utility class
    """
    def __init__(self, vec):
        self.vec = vec
    def __str__(self):
        return "%d dimensional vector" % len(self.vec)
    def __repr__(self):
        return "%d dimensional vector" % len(self.vec)
    def __len__(self):
        return len(self.vec)
