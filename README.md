# Explaining Decision Boundaries of CNNs Through Prototypes, Near Hits and Near Misses:
## A Comparison on Multiclass Medical Image Data Utilizing LRP Heatmaps
This is the repository to the master thesis of Bianca Zimmer

You can find the full code here. For any questions please contact
bianca-katharina.zimmer@stud.uni-bamberg.de
or anyone at the institution of Cognitive Systems at University of Bamberg

---

## Abstract
Deep Neural Networks (DNNs) are often criticized for being a so-called black
box. There have been many attempts to visualize the decision process of classification
DNNs. One of them is the approach by Herchenbach et al. (2022) to
give the domain expert an explanation of the decision boundaries by providing
prototypical images, Near Hit, and Near Miss images (P+NHNM). This thesis
aims to expand their approach to multi-class classification as well as basing
the image selection not only on feature embeddings (FE) but additionally on
heatmaps generated with layer-wise relevance propagation (LRP) (Lapuschkin
et al., 2015). For the selection of the P+NHNM images we used the MDD-critic
(Borgwardt et al., 2006), the Euclidean Distance, structural similarity
(SSIM) (Wang et al., 2004), and complex wavelet structural similarity (CW-
SSIM) (Sampat et al., 2009). As models, we set up a custom CNN as well as
an off-the-shelf VGG16 (Russakovsky et al., 2014).

We evaluated our findings on the MNIST dataset (Y. LeCun and Burges, 1998) and
the OCT dataset (Kermany et al., 2018a) using quantitative evalua
tion via comparisons of distance values, the Jaccard Index (Jaccard, 1901) and
the 1-NN classifier (Bien and Tibshirani, 2011). The qualitative evaluation consisted
of reviewing the selected P+NHNM images. Our results were not always
in line with those of Herchenbach et al. (2022) indicating that the off-the-shelf
VGG16 provided a good basis for the P+NHNM selection and often even a better one
than the custom CNN. We propose to try a transfer-learned CNN model
next. No clear recommendation for or against the LRP heatmaps as reference
point can be made, since our results show evidence for both cases, depending
on the underlying distance measure and dataset. The computational time for
the LRP heatmaps as well as the selection of NHNM images was several times
higher than for the FE and the corresponding NHNM images, suggesting that
FE might be the better choice when aiming for real-time applications.

---

## Folders:

### documentation
Folder with process flow images, class diagram and some additional information in the README.md.
Documentation for the master thesis.

### main_code
Is the modified code provided by Marvin Herchenbach et al. in the paper "Explaining Image
Classifications with Near Misses, Near Hits and Prototypes: Supporting Domain Experts in
Understanding Decision boundaries".
It came with a README.md file which I took the liberty to update and expand according to my needs.
This is the main code I used for conducting my master thesis. The main changes are:
1. Added ability to cope with a multi-class data set (as opposed to the original binary classification)
You have to set this ability in the utils.py file with the BINARY switch
2. Added ability to generate LRP heatmaps via the iNNvestigate library and calculate Near Hits
and Near Misses on their basis
3. Changed the original CNN so it would be able to train on the dataset of Kermany et. al (Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2)
4. Deleted the XAI demonstrator since it was not compatible with the tensorflow versions anymore

See own README.md in this folder for detailed information

### results
Contains some of our results used for the evaluation of the master thesis.
You can find example images of the prototypes as well as selected Near Hits and Near Misses (NHNM)
in the respective sub-folders.

### sort_ttv.py
Stands for _sort test train validation_. This is a helper Python file which can sort images
into the correct folder structure for code_mh_main to sue.
There is a little "How To" as a comment at the  beginning of the file

---

## Best practice:
To get the same results as in the master thesis you can follow the steps in 
_main_code/documentation/README.md_

The general workflow is also represented as flow charts in _code_mh_main/documentation/_

You can follow these steps if you want to use your own dataset:

1. Sort images via sort_ttv.py into the appropriate folders.
`python3 sort_ttv.py <from imagedir> <to basedir> <testsplit> -s <validationsplit>`
2. Change the _code_mh_main/utils.py_ file according to your needs
3. Run `python3 code_mh_main/preprocessing.py` and follow the instructions on the command line
4. Run `python3 code_mh_main/jaccard_evaluation_final.py` to view NHNMs of test images

Be aware that when running _main_code/preprocessing.py_ there is no possibility to test any
parameters. You should do this beforehand. E.g. find the best parameter for the number of prototypes
via screeplot/ellbow method in the _main_code/prototype_selection.py_ file.
Same goes for all tweaking parameters for LRP and of course training parameters for the CNN.

## Citation
None available yet

Zimmer, Bianca (2023), “Explaining Decision Boundaries of CNNs Through Prototypes, Near Hits and Near Misses:
A Comparison on Multiclass Medical Image Data Utilizing LRP Heatmaps”, University of Bamberg, Master Thesis

## Licence
CC BY 4.0

https://creativecommons.org/licenses/by/4.0/
