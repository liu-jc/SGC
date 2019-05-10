
## Tasks 
1. concatenate different features with different alpha and see if it can improve the performance. (Worse results)

2. After getting the feature vectors, multiply the feature vectors with their degree. (It may be done in matrix form) (Worse results)

## Degree results
1. cora 0.8141(degree 20, RWalk), 0.8075(degree 5, AugNorm)
2. citeseer 0.6861(degree 32, RWalk), 0.6803(degree 8, AugNorm)
3. pubmed 0.7813(degree 30, RWalk),0.7724 (degree 6, AugNorm)

## Multi_scale (degree = 2)
1. cora 0.8015(RWalk w), 0.7899(RWalk w/o),0.7925(AugNorm)
2. citeseer 0.6809(RWalk w), 0.6741(RWalk w/o), 0.6715(AugNorm)
3. pubmed 0.7722(RWalk w), 0.7667(RWalk w/o), 0.7675(AugNorm)
   
## Multiply degree 
1. cora 0.7863(RWalk w)
2. citeseer 0.6605(RWalk w)
3. pubmed 0.7692(RWalk w)

# TODO 
1. check if any features of nodes are very small during the propagation, if so, try to fix them (or do not go to these nodes)
2. change the scale, try to not use inverse. If you inverse the matrix, after a few iteration, it will become very small. Can we use another way? 
3. check renchi 