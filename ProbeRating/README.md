## Source Codes

### Usage
Examples to run ProbeRating and FastBioseq with toy sample data. Please also see the *Implementation Details* section below for the Python version and dependent packages.

1. Train FastBioseq embedding model. 
   - e.g.:
```python
python train_FastBioseq.py sample_data/sample1.fa sample_data/trained_FastBioseq_models/sample1_model 10 3 2
```

2. Generate embedding vectors for biological sequences, using pre-trained FastBioseq model. 
   - e.g.:
```python
python genVec.py sample_data/trained_FastBioseq_models/sample2_model sample_data/sample1.fa sample1_FT.csv 2
```

3. Run the ProbeRating recommender.
   - e.g. (for RRM; HOMEO is similar):
```python
python probeRating_recommender_nn_RRM.py 1 2 0.1 30 0 2 0.01 0 1 sample_data/sample4.mat sample_data/sample3_FT.csv 0 tanh 0.5 3 10 10
```
-----

### Implementation Details
* We developed ProbeRating’s FastBioseq on top of the implementation of FastText, Doc2Vec, and Word2Vec from the Genism python library (Rehurek and Sojka, 2010), and the implementation of ProtVec (Asgari and Mofrad, 2015). Python packages used:
	- Python 2.7 (also work for Python 3 with few updates in the syntax)
	- numpy 1.15.2
	- pandas 0.23.3
	- scipy 1.1.0
	- biopython 1.72
	- gensim 3.6.0
	- memory-profiler 0.54.0
	- guppy 0.1.10

* We developed ProbeRating’s neural network recommender using Tensorflow and Keras libraries (Abadi et al., 2015).  Python packages used:
	- Python 3.5
	- numpy 1.16.0
	- pandas 0.23.3
	- scipy 1.1.0
	- biopython 1.72
	- h5py 2.8.0
	- scikit-learn 0.20.0
	- keras 2.2.4
	- tensorflow 1.11.0

* Besides, we used a CentOS Linux 7 node with one NVIDIA Tesla P100 GPU, three Intel Xeon E5-2683 CPUs, and 12 GB RAM, for the paper.
-----

### References
> Abadi, M. et al. (2015), TensorFlow: Large-scale machine learning on heterogene-ous systems. Software available from tensorflow.org. https://www.tensor-flow.org/
>
> Asgari, E. and Mofrad, M. R. (2015), Continuous distributed representation of biological sequences for deep proteomics and genomics, PLoS One 10(11), e0141287. https://github.com/kyu999/biovec
>
> Rehurek, R. and Sojka, P. (2010), Software Framework for Topic Modelling with Large Corpora, In Proc LREC, pp. 45–50. https://radimrehurek.com/gensim/models/fasttext.html
