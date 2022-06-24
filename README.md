# Advanced Systems Lab - Fast and efficient data valuation - Team 12 [[PDF](https://gitlab.inf.ethz.ch/COURSE-ASL/asl22/team12/-/tree/master/report/12_report.pdf)]
In the field of interpretable artificial intelligence the calculation of shapley values has become increasingly important. Since calculating such shapley values is computationally very expensive, having a high-performance implementation is fundamental. Recently Jia et al. [1] introduces two algorithms with a significant lower runtime complexity for calculating shapley values in the case of a K-nearset neighbour models. In this paper we take these two algorithms as a baseline and present for both a highly-optimized single-core implementation. Through cache locality optimization, increase of instruction level parallelism, vectorization and various other optimization techniques we achieve a speedup of 25x in respect to the original python implementation

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