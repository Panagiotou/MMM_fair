### This is MMM-fair classifier, originally inspired from the paper "Multi-fairness Under Class-Imbalance"
https://link.springer.com/chapter/10.1007/978-3-031-18840-4_21
### Original published work was meant for working with only Equalized Odds (or Disperate Mistreatment). This implementation covers different fairness objectives namely Equalized Odds, Demographic Parity, Equal Oppostunity. 
# 
### Different fairness objectives need changes in mmm-cost calculation. Further, we strenthened alpha (weight of each individual learners) calculation directly with a fairness weight, replaced weak learners with stronger (Decision tree with depth>1) learners, and dynamically also remove over boosted samples (a problem identified in the original MMM paper). 