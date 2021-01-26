# EVT_POT
Peak Over Threshold POT estimator

It seems that Peak Over Threshold has become popular way to calculate thresholds for anomalies. Having some distribution, high empirical quantile can be used to cut samples which are at the tail of the distribution. These values, above this quantile follow GPD generalized Pareto distribution. GPD parameters sigma and gamma can be estimated and used to calculate threshold for some low probability that any sample gets above calculated threshold

papers used 

1. https://hal.archives-ouvertes.fr/hal-01640325/document
2. https://www.researchgate.net/publication/245293243_An_Application_of_the_Peaks_Over_Threshold_Method_to_Predict_Extremes_of_Significant_Wave_Height
3. https://www.researchgate.net/publication/334717291_Robust_Anomaly_Detection_for_Multivariate_Time_Series_through_Stochastic_Recurrent_Neural_Network
4. https://www.hindawi.com/journals/complexity/2020/8846608/

approach

1. haircut data. use high empirical quantile like 98th percentile . use loss data from my mtad-tf repository - implementation of papaer 4. this haircut follows GPD. see all four papers discussing initial haircut threshold
2. GPD maximum likelihood estimation method provides log likelihood to maximize. see paper 1 on page 3. tensorflow is used to minimize negative of that. this gives us estimates for sigma and gamma for GPD. Funny thing, it gets better log compare to omnianomaly code.
3. use equation 1 from paper 1 to calculate threshold. decide on probability that values of distribution will not pass threshold. see paper 3 and 4 suggesting probability value


My setup:
- I use LOSS from mtad-tf project as input data
- I use GCP to store data and models

Steps

1. Calculate 98% quantile
python quantile.py --value_path=RMS_loss.csv --quantile=0.98

2. Create tfrecords file. RMS_loss.csv values are cut with recentile value 
python prepare_data.py --files_path=RMS_loss.csv --tfrecords_file=gs://anomaly_detection/pot/data/train/{}-0.98.tfrecords --t=0.9203352009999999

3. Train model . Minimize log likelihood , use negative of that formula. Batch size must be the same as whole input tfrecords file since it is extected by log likelihood formula
python training.py --action=TRAIN --train_file=gs://anomaly_detection/pot/data/train/loss_machine-1-1-0.98.tfrecords --output_dir=gs://anomaly_detection/pot/output/machine-1-1.0.98 --n=28378 --t=0.9203352009999999 --q=0.001 --num_train_steps=2000 --batch_size=568 --clip_gradients=1.0 --learning_rate=0.001

4. Extract estimates of gamma, sigma and calculate threshold at q probability
python training.py --action=PREDICT --train_file=gs://anomaly_detection/pot/data/train/loss_machine-1-1-0.98.tfrecords --output_dir=gs://anomaly_detection/pot/output/machine-1-1.0.98 --n=28378 --t=0.9203352009999999 --q=0.001 --batch_size=568
