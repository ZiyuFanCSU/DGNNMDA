# DGNNMDA
Predicting miRNA-drug sensitivity associations based on dual channel graph neural network via heterogeneous information network.

The programs is supported by Python 3.7.4 and Tensorflow 1.14+. In sample and data directories, the function of each script and data file is briefly described.
# Input
* miRNA-drug sensitivity association network
* miRNA-miRNA association network
* drug-drug associations network
# Method
By calculating the similarity between miRNA-miRNA, drug-drug, we constructed the heterogeneous information network of miRNA and drug, and designed the implemented, simplified version versus improved
version of GNN model for information transfer between the homogeneous nodes and heterogeneous nodes, respectively, to obtain high-quality embedded expression.
# Usage
* 1.Configure the xx.conf file in the directory named config. (xx is the name of the model)
* 2.Dividding the dataset into train and test sets, and placed in the corresponding 'train.txt' and 'test.txt' positions in the code. Common methods include 5-fold cross-validation, etc. You have access to sklearn.model_selection.KFold.
* 3.Run main.py.
