# Implementing fast vectorized single-core shapley value calculation [[PDF](https://github.com/municola/fast-shapley-values/edit/master/12_report.pdf)]

## Dataset
As described by Jia et al. [1] we extract the features from ResNet and save them as binaries. Download the already prepared dataset her:
[Cifar10](https://polybox.ethz.ch/index.php/s/flCES6dSsSL7LKD)

## Example usage
1. ```git clone git@gitlab.inf.ethz.ch:COURSE-ASL/asl22/team12.git```
2. ```cd team12/code```
3. ```make```
4. ```./shapley_values```

## References:
[1] Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nezihe Merve Gurel, Bo Li, Ce Zhang, Costas J Spanos, and Dawn Song, “Efficient task-specific data valuation for nearest neighbor algorithms,” arXiv preprint arXiv:1908.08619, 2019
