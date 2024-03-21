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
* 2.The file 'train.txt' and 'test.txt' are divided based on the dataset using 5-fold cross-validation.
* 3.Run main.py.
# References
If you use our repository, please cite the following related paper:

```
@article{deng2022dual,
  title={Dual-channel heterogeneous graph neural network for predicting microRNA-mediated drug sensitivity},
  author={Deng, Lei and Fan, Ziyu and Xiao, Xiaojun and Liu, Hui and Zhang, Jiaxuan},
  journal={Journal of Chemical Information and Modeling},
  volume={62},
  number={23},
  pages={5929--5937},
  year={2022},
  publisher={ACS Publications}
}
```
