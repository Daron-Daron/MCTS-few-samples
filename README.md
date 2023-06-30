# MCTS-few-samples
Adaptation of Dilina's https://github.com/dilina-r/ to when few group-item samples are available.

The main algorithm is PriorsGoodReads4Clusters. It runs the new algorithm on the Goodreads data set with 4 clusters, and 10 samples

It loads the data files mu_goodreads4.csv and sigma_goodreads4.csv which are the means and variances of the group-item matrix. These are used to draw new ratings from simulated users.

For the UCB step we use the empirical update (19) in the attached notes UniformPrior2.pdf.  The integrals are computed offline and stored in  Update_factors_GR_4_clusters_10_samples. There is one update factor per group-item-sample. The samples are drawn from the true means and variances. But after that , we only use the true means and variances to draw 

 The images are generated from PriorsGoodReads4Clusters.They show performance of the algorithm for 20 users per group.
