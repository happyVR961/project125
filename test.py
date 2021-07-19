import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
y = pd.read_csv("project122.csv")["labels"]
x = np.load("image.npz")["arr_0"]
print(pd.Series(y).value_counts())
classes = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)
print(nclasses)
xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size = 2500, train_size = 7500)
xTrainScaled = xTrain/255.0
xTestScaled = xTest/255.0
lr = LogisticRegression(solver = "saga",multi_class="multinomial").fit(xTrainScaled, yTrain)
def getPrediction(image):
    impil = Image.open(image)
    imagebw = impil.convert("L")
    imageresized = imagebw.resized((28,28), Image.ANTIALIAS)
    pixelfilter = 20
    minpixel = np.percentile(imageresized, pixelfilter)
    imageinverted = np.clip(imageresized - minpixel, 0, 255)
    maxpixel = np.max(imageresized)
    imageinverted = np.asarray(imageinverted)/maxpixel
    testsample = np.array(imageinverted).reshape(1,784)
    testpredict = lr.predict(testsample)
    return testpredict[0]
    