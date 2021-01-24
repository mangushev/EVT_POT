# EVT_POT
Peak Over Threshold POT estimator

It seems that Peak Over Threshold has become popular way to calculate thresholds for anomalies. Having some distribution, high empirical quantile can be used to cut samples which are at the tail of the distribution. These values, above this quantile follow GPD generalized Pareto distribution. GPD parameters sigma and gamma can be estimated and used to calculate threshold for some low probability that any sample value less than calculated threshold

this is draft material. 
