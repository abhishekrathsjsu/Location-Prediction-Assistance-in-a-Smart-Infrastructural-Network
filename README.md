# Smart Network Location Prediction Assistant

The assistant program predicts the user's potential location based on previous locations to be used for Smart Home Network.

## Dataset

You can use the dataset from the file indoor_movement.csv in the repository or visit https://archive.ics.uci.edu/ml/datasets/Indoor+User+Movement+Prediction+from+RSS+data.

## Project Info
The project is focused on classification work to predict how users are moving in real-life smart home environments using the sensors from Smart Home Network without the usage of Neural Network. Input data comprises temporary ranges of radio signal intensity provided by five sensors: four ambient anchors and one user-sized mote. The project consists of three area environments, composed of two traditional furniture rooms divided by a hallway. 

## Model Info
Libraries used: numpy, pandas, sklearn, plotly
Language used: Python
Models used: Random Forest Classifier, Extra Tree Classifier, Bagging Classifier, AdaBoost Classifier, Gradient Boost Classifier, KNN and Logistic Regression

Improved model to attain 73% precision, 73% recall and 0.73 recall by changing positioning system’s hyperparameters using Extra Tree Classifier.

## Note
Suggestions are welcome to improve this project.
