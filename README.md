A simple example that shows how to implement your own simple linear regression using TensorFlow in Python.
Below is an overiew about the existing files:
* Points.csv: Data file that will be used to train and test our model
* DataHelper.py: A helper script that:
  * Reads the data file
  * Separate data from labels
  * Split into training and test using sklearn.cross_validation
* MyLinearRegression.py: The main script that contains the actual model. Once the script executes, it creates a directory called 'tf_model' that contais:
  * checkpoint
  * model.ckpt.data-00000-of-00001
  * model.ckpt.index
  * model.ckpt.meta
  * model.pbtxt
  * frozen.pb
  * optimized.pb
