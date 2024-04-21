# HCR using the MINST dataset of keras
# a dataset of 60,000 28x28 grayscale images of the 10 digits, along with
# a test set of 10,000 images.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
data = fetch_openml("mnist_784", version=1)
print(data)

x, y = data["data"], data["target"]
print(x.shape)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2,random_state=42)

image = np.array(xtrain.iloc[0]).reshape(28, 28)
plt.imshow(image)

from sklearn.linear_model import SGDClassifier
model = SGDClassifier()
model.fit(xtrain, ytrain)

predictions = model.predict(xtest)
print(predictions)
# Chk in ['8' '4' '8' ... '2' '7' '1']
image = np.array(xtest.iloc[0]).reshape(28, 28)
plt.imshow(image)

image = np.array(xtest.iloc[1]).reshape(28, 28)
plt.imshow(image)

