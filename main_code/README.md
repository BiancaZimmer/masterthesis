## Structure

```  

├── classify.py
├── dataentry.py
├── dataset.py
├── feature_extractor.py
├── helpers.py
├── helpers_innvestigate.py
├── jaccard_evaluation_distance_comparison.ipynb
├── jaccard_evaluation_final.ipynb
├── kernels.py
├── licenses.rst
├── LRP_heatmaps.py
├── mmd_critic.py
├── modelsetup.py
├── near_miss_hits_selection.py
├── offline_feat_extraction.py
├── pickle_preprocessing.py
├── prototype_selection.py
├── README.md
├── requirements.txt            || Apply to avoid any dependency errors 
├── static                      || Folder for static files - e.g. data, embeddings, models
├── utils.py

```

## Data (Not included on GitHub)
- Original Code: https://gitlab.cc-asp.fraunhofer.de/sees/vis-ml2022-mh Herchenbach, Marvin & Müller, Dennis & Scheele, Stephan & Schmid, Ute. (2022). Explaining Image Classifications with Near Misses, Near Hits and Prototypes: Supporting Domain Experts in Understanding Decision Boundaries. 10.1007/978-3-031-09282-4_35. 
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- OCT Dataset: https://data.mendeley.com/datasets/rscbjbr9sj/2? Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2

---

## How To

1. Sort images `./sort_ttv.py`
2. Change `utils.py` according to your needs
3. Cut images into square format `helpers.py`
4. Train Model `modelsetup.py`
5. Create Feature Embeddings `dataentry.py`
6. Create LRP Heatmaps `LRP_heatmaps.py`
7. Create Prototypes `prototype_selection.py`
8. Generate Near Hits & Near Misses `near_miss_hits_selection.py`
9. Evaluation ``pickle_preprocessing.py`` and ``jaccard_evaluation_final.py``
10. View NHNM in ``jaccard_evaluation_final.py``

Steps 2.-6. can be done via `preprocessing.py` if all parameters have been tried and tested before

To get the same results as in the master thesis you can follow the steps in 
`documentation/README.md`

The general workflow is also represented as flow charts in _documentation/_

