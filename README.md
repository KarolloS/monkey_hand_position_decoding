# Hand position decoding
Predicting monkey's hand trajectory based on the neural activity. Monkey performed eight reaching movements at different angles.
During each movement neural activity in the form of spikes from 98 neurons were recorded. The task is to estimate planar (X,Y) 
position of the monkey's hand during movement based only on actual and historical neural activity.

## Estimate direction
In order to estimate the direction of the movement, artificial neural networks were used. Performance was improved by using committee 
machine of 3 networks. Data preparation was identified as a key problem - in order to reduce the size of the input, temporal average 
of spikes was calculated. Achieved accuracy equals **98%**, training and testing took ~30s. Confusion matrix:
![alt text](https://github.com/KarolloS/monkey_hand_position_decoding/blob/master/confusion_matrix.png)

## Estimate trajectory
Average trajectory was calculated for each movement direction. In order to estimate hand position, one of the eight average trajectories 
were chosen based on the direction estimation. Then, for each time step X and Y was obtained directly from average trajectory. RMSE 
equals **9.161mm**. All trajectories (actual and average):
![alt text](https://github.com/KarolloS/monkey_hand_position_decoding/blob/master/trajectories.png)

## Code
File `monkeydata_training.mat` contains all available data. `positionEstimatorTraining.m` and `positionEstimator.m` train and test the 
estimator respectively. Data training/testing division is performed in script `testFunction.m`.

During initial attempts of direction decoding Python and TensorFlow were used. Implementation of neural network which leads to classification
accuracy of about 97% can be found in `BMI.py`.

Project was done as a part of Brain-Machine Interfaces course at Imperial College London. This estimator won the **first prize** during internal
competition as its performance was the best among other estimators programmed by other students.
