### ProbeRating: A recommender system to infer binding profiles for nucleic acid-binding proteins

Welcome!

This repo stores codes and data for the paper:
> Shu Yang, Xiaoxi Liu, Raymond Ng. ProbeRating: A recommender system to infer binding profiles for nucleic acid-binding proteins. 

It contains: 
1. Codes for the biological sequence embedding package FastBioseq.

Usage:
```python
python train_FastBioseq.py example1.fa trained_models/example1_model 25 3 2
python genVec.py trained_models/example1_model example2.fa example2_FT.csv
```
2. Codes for the neural network based recommender system used in ProbeRating
* The data used in the paper were downloaded from public databases and published papers: 
   * CISBP database <http://cisbp.ccbr.utoronto.ca/>
   * CISBP-RNA database <http://cisbp-rna.ccbr.utoronto.ca/>, and its associated paper <http://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/>
   * Affinity Regression paper <https://www.nature.com/articles/nbt.3343>
   * Uniprot database <https://www.uniprot.org/>



