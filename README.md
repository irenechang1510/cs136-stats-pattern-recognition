# [CS136](https://www.cs.tufts.edu/comp/136/2022s/) - Stats Pattern Recognition Projects

[Coding Project 01](cp1-unigram-probabilities): unigram model, addressing two key questions:
- Given training data, how should we estimate the probability of each word? How might estimates change if we have very little (or abundant) data?
- How can we select hyperparameter values to improve our predictions on heldout data, using only the training set?

[Coding Project 02](cp2-bayesian-linreg): implement the Bayesian linear regression model and Bayesian logistic regression model, addressing two key questions:
- Given training data, how should we estimate the probability of a new observation? How might estimates change if we have very little (or abundant) training data?
- How can we select hyperparameter values to improve our predictions on heldout data? How can we contrast methods like cross validation with methods that might use our whole training set more effectively?

[Coding Project 03](cp3-mcmc): investigate detailed balance for the Metroplois-Hastings algorithm, and implement 2 possible algorithms for sampling from a target distribution: the Metropolis algorithm and the Gibbs Sampling algorithm. The homework problem addresses the theoretical properties of MCMC sampling. The coding problems will try to address several key practical questions:
- How can we use MCMC methods to sample from target distributions?
- What are the tradeoffs of using Metropolis sampling vs. Gibbs sampling? (runtime, implementation time, reliability, tunability, etc.)
- How do we effectively implement MCMC methods in practice? (e.g. selecting hyperparameters, deciding how many samples to draw, etc.)

[Coding Project 04](cp4-gaussian-mixture-models): compare 2 possible algorithms for learning the parameters (weights, means, and standard-deviations) of a Gaussian mixture model: LBFGS Gradient descent, Expectation Maximization. The problems below will try to address several key practical questions:
- How sensitive are methods to initialization?
- How can we effectively select the number of components?
- What kind of structure can GMMs uncover within a dataset?

[Coding Project 05](cp5-hidden-markov): implement 2 possible algorithms for Hidden Markov models: the Forward algorithm and the Viterbi algorithm. The problems will try to address several key practical questions:
- How can we use dynamic programming to accomplish these computations?
- How can I turn written mathematics into working NumPy code?

# CS119 - Big Data

[HW3](cs119-hw/CS119-Quiz3.pdf): Opioid Files with BigQuery

[HW4](cs119-hw/CS119-Quiz4.pdf): Traffic Rank algorithm, Hadoop Errors, Analyze valence in Presidents' speeches using map-reduce

[HW5](cs119-hw/CS119-Quiz5.pdf): Textual Processing using Hadoop

[HW6](cs119-hw/CS119-Quiz6.pdf): Data Streams (DStream), buy-sell stocks and bloom filter

[HW7](cs119-hw/CS119-Quiz7.pdf): Processing text (Sentence Similarity, Frequent Itemset Mining, NTLK, LDA on clothing review)
