import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris dataset
iris = load_iris()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Define feature columns
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

# Define DNNClassifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    n_classes=3,
    model_dir="iris_model"
)

# Define input functions
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=None,
    shuffle=True
)

test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False
)

# Train model
classifier.train(input_fn=train_input_fn, steps=2000)

# Evaluate model
eval_result = classifier.evaluate(input_fn=test_input_fn)

print("Test set accuracy: {accuracy:0.3f}".format(**eval_result))
