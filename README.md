# EVT_POT
Peak Over Threshold POT estimator

It seems that Peak Over Threshold has become popular way to calculate thresholds for anomalies. Having some distribution, high empirical quantile can be used to cut samples which are at the tail of the distribution. These values, above this quantile follow GPD generalized Pareto distribution. GPD parameters sigma and gamma can be estimated and used to calculate threshold for some low probability that any sample value less than calculated threshold

this is draft material. 

papers used 

1 https://hal.archives-ouvertes.fr/hal-01640325/document
2 https://www.researchgate.net/publication/245293243_An_Application_of_the_Peaks_Over_Threshold_Method_to_Predict_Extremes_of_Significant_Wave_Height
3 https://www.researchgate.net/publication/334717291_Robust_Anomaly_Detection_for_Multivariate_Time_Series_through_Stochastic_Recurrent_Neural_Network

approach

1. haircut data. use high empirical quantile above 90 percent. use loss data from my mtad-tf repository. this haircut follows GPD. see all three papers discussing initial haircut threshold
2. GPD maximum likelihood estimation method provides log likelihood to maximize. see paper 1 on page 3. tensorflow is used to minimize negative of that. this gives us estimates for sigma and gamma for GPD
3. use equation 1 from paper 1 to calculate threshold. decide on probability that values of distribution will not pass threshold. see paper 3 suggesting probability value

steps

1 prepare training data 
2 train model . gamma and sigma are trainable model parameters
3 run predict to extract gamma and sigma and calculate threshold
