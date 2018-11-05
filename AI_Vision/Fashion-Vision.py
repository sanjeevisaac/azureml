# Databricks notebook source
# MAGIC %md
# MAGIC 1. Add the following Python Libraries to the Databricks Workspace
# MAGIC     <code>
# MAGIC     - Click on "Workspace" > "Shared" > "Create Library" > "Upload Python Egg or PyPi" <br>
# MAGIC       azure-cognitiveservices-vision-customvision <br>
# MAGIC     - Create a new library from Maven coordinates in your workspace.<br>
# MAGIC       For the coordinates use: Azure:mmlspark:0.14
# MAGIC     </code>
# MAGIC     Select "Automatically attach to all clusters"<br>
# MAGIC     Ensure that your Spark cluster has at least Spark 2.1 and Scala 2.11.<br>
# MAGIC 2. Deploy Bing Search API, Custom Vision API services using the Azure Portal (portal.azure.com)<br>
# MAGIC    Fetch the Bing Search Key, Custom Vision Training Key, and Custom Vision Prediction Key from the corresponding services and paste them in the textboxes at the top.

# COMMAND ----------

dbutils.widgets.text("bing_search_api_key", "", "Bing Search API Key:")
dbutils.widgets.text("custom_vision_training_key", "", "Custom Vision Training Key:")
dbutils.widgets.text("custom_vision_prediction_key", "", "Custom Vision Prediction Key:")

# COMMAND ----------

from mmlspark import *
from mmlspark import FluentAPI
import os
from pyspark.sql.functions import lit

BING_IMAGE_SEARCH_KEY = dbutils.widgets.get("bing_search_api_key")
def bingPhotoSearch(name, queries, pages):
  offsets = [offset*10 for offset in range(0, pages)] 
  parameters = [(query, offset) for offset in offsets for query in queries]
  
  return spark.createDataFrame(parameters, ("queries","offsets")) \
    .mlTransform(
      BingImageSearch()                             # Apply Bing Image Search
        .setSubscriptionKey(BING_IMAGE_SEARCH_KEY)  # Set the API Key
        .setOffsetCol("offsets")                    # Specify a column containing the offsets
        .setQueryCol("queries")                     # Specify a column containing the query words
        .setCount(10)                               # Specify the number of images to return per offset
        .setImageType("photo")                      # Specify a filter to ensure we get photos
        .setOutputCol("images")) \
    .mlTransform(BingImageSearch.getUrlTransformer("images", "urls")) \
    .withColumn("labels", lit(name)) \
    .limit(200)

def displayDF(df, n=5, image_cols = set(["urls"])):
  rows = df.take(n)
  cols = df.columns
  header = "".join(["<th>" + c  + "</th>" for c in cols])
  
  style = """
<!DOCTYPE html>
<html>
<head>
<style>
table {
    font-family: arial, sans-serif;
    border-collapse: collapse;
    width: 300;
}

td, th {
    border: 1px solid #dddddd;
    text-align: left;
    padding: 8px;
}

tr:nth-child(even) {
    background-color: #dddddd;
}
</style>
</head>"""
  
  table = []
  for row in rows:
    table.append("<tr>")
    for col in cols:
      if col in image_cols:
        rep = '<img src="{}",  width="100">'.format(row[col])
      else:
        rep = row[col]
      table.append("<td>{}</td>".format(rep))
    table.append("</tr>")
  tableHTML = "".join(table)
  
  body = """
<body>
<table>
  <tr>
    {} 
  </tr>
  {}
</table>
</body>
</html>
  """.format(header, tableHTML)
  try:
    displayHTML(style + body)
  except:
    pass

# COMMAND ----------

from azure.cognitiveservices.vision.customvision.training import training_api
from azure.cognitiveservices.vision.customvision.training.models import ImageUrlCreateEntry

CUSTOM_VISION_TRAINING_KEY = dbutils.widgets.get("custom_vision_training_key")
CUSTOM_VISION_PREDICTION_KEY = dbutils.widgets.get("custom_vision_prediction_key")

trainer = training_api.TrainingApi(CUSTOM_VISION_TRAINING_KEY)
projects = trainer.get_projects()
tag_dict = {}
project = next((project for project in projects if project.name == "ShoeStyleTagger"), None)
if project == None:
  # Create a new project
  project = trainer.create_project("ShoeStyleTagger")
  tag_dict["sandals"] = trainer.create_tag(project.id, "sandals")
  tag_dict["slippers"] = trainer.create_tag(project.id, "slippers")
  tag_dict["sneakers"] = trainer.create_tag(project.id, "sneakers")
  tag_dict["boots"] = trainer.create_tag(project.id, "boots")
else:
  tags = trainer.get_tags(project.id)
  tag_dict = dict(((tag.name, tag) for tag in tags))

# COMMAND ----------

tag_dict

# COMMAND ----------

sandalsQueries = ["site:bloomingdales.com Womens Sandals"]
sandalsUrls = bingPhotoSearch(tag_dict["sandals"].id, sandalsQueries, pages=100)
displayDF(sandalsUrls)

# COMMAND ----------

slippersQueries = ["site:bloomingdales.com Womens Slippers"]
slippersUrls = bingPhotoSearch(tag_dict["slippers"].id, slippersQueries, pages=100)
displayDF(slippersUrls)

# COMMAND ----------

sneakersQueries = ["site:bloomingdales.com Womens Sneakers"]
sneakersUrls = bingPhotoSearch(tag_dict["sneakers"].id, sneakersQueries, pages=100)
displayDF(sneakersUrls)

# COMMAND ----------

bootsQueries = ["site:bloomingdales.com Womens Boots"]
bootsUrls = bingPhotoSearch(tag_dict["boots"].id, bootsQueries, pages=100)
displayDF(bootsUrls)

# COMMAND ----------

imageUrls = sandalsUrls.union(slippersUrls).union(sneakersUrls).union(bootsUrls).repartition(2)\
  .dropna()

# COMMAND ----------

imageUrls.count()

# COMMAND ----------

from pyspark.ml import Pipeline, Transformer
from typing import Iterable
# CUSTOM TRANSFORMER
class UploadToCustomVision(Transformer):
    """
    A Custom Transformer which uploads the images to Custom Vision Project along with the tags
    """

    def __init__(self, project_id=None):
        super(UploadToCustomVision, self).__init__()
        self.project_id = project_id

    def _transform(self, df: DataFrame) -> DataFrame:
        imagesDF = df.toPandas().set_index('urls').T.to_dict('list')
        for img in imagesDF:
          product_img_link = img
          product_img_link = product_img_link.replace("bloomingdales.com", "bloomingdalesassets.com")
          tagList = imagesDF[img]
          summary = trainer.create_images_from_urls(self.project_id, [ImageUrlCreateEntry(url=product_img_link,tag_ids=tagList)])
        return df


# COMMAND ----------

custom_vision_upload = UploadToCustomVision(project_id = project.id)
df_with_result = custom_vision_upload.transform(imageUrls)

# COMMAND ----------

import time

print ("Training...")
iteration = trainer.train_project(project.id)
while (iteration.status != "Completed"):
    iteration = trainer.get_iteration(project.id, iteration.id)
    print ("Training status: " + iteration.status)
    time.sleep(1)

# The iteration is now trained. Make it the default project endpoint
trainer.update_iteration(project.id, iteration.id, is_default=True)
print ("Done!")

# COMMAND ----------

# MAGIC %md
# MAGIC Try this image for prediction:<br> "https://tse2.mm.bing.net/th?id=OIP.QpxuXic1_vcHKsSO7cwkQwHaHa&pid=Api"<br>
# MAGIC <img src="https://tse2.mm.bing.net/th?id=OIP.QpxuXic1_vcHKsSO7cwkQwHaHa&pid"></img>

# COMMAND ----------

from azure.cognitiveservices.vision.customvision.prediction import prediction_endpoint
from azure.cognitiveservices.vision.customvision.prediction.prediction_endpoint import models

# Now there is a trained endpoint that can be used to make a prediction

predictor = prediction_endpoint.PredictionEndpoint(CUSTOM_VISION_PREDICTION_KEY)

test_img_url = "https://tse2.mm.bing.net/th?id=OIP.QpxuXic1_vcHKsSO7cwkQwHaHa&pid=Api"
results = predictor.predict_image_url(project.id, iteration.id, url=test_img_url)

# Display the results.
for prediction in results.predictions:
    print ("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))

# COMMAND ----------

# MAGIC %md
# MAGIC Export the model to run on edge/mobile: <br>
# MAGIC https://github.com/MicrosoftDocs/azure-docs/blob/master/articles/cognitive-services/Custom-Vision-Service/export-model-python.md

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /FileStore/tables/model.pb 

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /FileStore/tables/labels.txt 

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -lt /dbfs/FileStore/tables/

# COMMAND ----------

import tensorflow as tf
import os

filename = '/dbfs/FileStore/tables/model.pb'
labels_filename = '/dbfs/FileStore/tables/labels.txt'

graph_def = tf.GraphDef()
labels = []

# Import the TF graph
with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Create a list of labels.
with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

# COMMAND ----------

def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

# COMMAND ----------

from PIL import Image
import numpy as np
import cv2

# Load from a file
imageFile = "/dbfs/FileStore/tables/testshoe.jpg"
image = Image.open(imageFile)

# Update orientation based on EXIF tags, if the file has orientation info.
image = update_orientation(image)

# Convert to OpenCV format
image = convert_to_opencv(image)

# COMMAND ----------

# If the image has either w or h greater than 1600 we resize it down respecting
# aspect ratio such that the largest dimension is 1600
image = resize_down_to_1600_max_dim(image)
# We next get the largest center square
h, w = image.shape[:2]
min_dim = min(w,h)
max_square_image = crop_center(image, min_dim, min_dim)
# Resize that square down to 256x256
augmented_image = resize_to_256_square(max_square_image)
# The compact models have a network size of 227x227, the model requires this size.
network_input_size = 227

# Crop the center for the specified network_input_Size
augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

# COMMAND ----------

# These names are part of the model and cannot be changed.
output_layer = 'loss:0'
input_node = 'Placeholder:0'

with tf.Session() as sess:
    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    predictions, = sess.run(prob_tensor, {input_node: [augmented_image] })

# COMMAND ----------

# Print the highest probability label
highest_probability_index = np.argmax(predictions)
print('Classified as: ' + labels[highest_probability_index])

# Or you can print out all of the results mapping labels to probabilities.
label_index = 0
for p in predictions:
    truncated_probablity = np.float64(np.round(p,8))
    print (labels[label_index], truncated_probablity)
    label_index += 1