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
project = next((project for project in projects if project.name == "ShoeStyleTagger"), None)
if project == None:
  # Create a new project
  project = trainer.create_project("ShoeStyleTagger")

# COMMAND ----------

style_tag_sandals = trainer.create_tag(project.id, "sandals")
style_tag_slippers = trainer.create_tag(project.id, "slippers")
style_tag_sneakers = trainer.create_tag(project.id, "sneakers")
style_tag_boots = trainer.create_tag(project.id, "boots")

# COMMAND ----------

sandalsQueries = ["site:bloomingdales.com Womens Sandals"]
sandalsUrls = bingPhotoSearch(style_tag_sandals.id, sandalsQueries, pages=100)
displayDF(sandalsUrls)

# COMMAND ----------

slippersQueries = ["site:bloomingdales.com Womens Slippers"]
slippersUrls = bingPhotoSearch(style_tag_slippers.id, slippersQueries, pages=100)
displayDF(slippersUrls)

# COMMAND ----------

sneakersQueries = ["site:bloomingdales.com Womens Sneakers"]
sneakersUrls = bingPhotoSearch(style_tag_sneakers.id, sneakersQueries, pages=100)
displayDF(sneakersUrls)

# COMMAND ----------

bootsQueries = ["site:bloomingdales.com Womens Boots"]
bootsUrls = bingPhotoSearch(style_tag_boots.id, bootsQueries, pages=100)
displayDF(bootsUrls)

# COMMAND ----------

sandalsDF = sandalsUrls.toPandas().set_index('urls').T.to_dict('list')
slippersDF = slippersUrls.toPandas().set_index('urls').T.to_dict('list')
sneakersDF = sneakersUrls.toPandas().set_index('urls').T.to_dict('list')
bootsDF = bootsUrls.toPandas().set_index('urls').T.to_dict('list')

# COMMAND ----------

bootsDF

# COMMAND ----------

from itertools import islice
product_img_links = list(islice(sneakersDF, 64))
for sneakers in list(islice(sneakersDF, 64)):
    product_img_link = sneakers
    product_img_link = product_img_link.replace("bloomingdales.com", "bloomingdalesassets.com")
    tagList = sneakersDF[sneakers]
    summary = trainer.create_images_from_urls(project.id, [ImageUrlCreateEntry(url=product_img_link,tag_ids=[style_tag_sneakers.id])])
    print(summary.is_batch_successful, product_img_link)

# COMMAND ----------

from itertools import islice
product_img_links = list(islice(bootsDF, 64))
training_images = []
for boots in list(islice(bootsDF, 64)):
    product_img_link = boots
    product_img_link = product_img_link.replace("bloomingdales.com", "bloomingdalesassets.com")
    tagList = bootsDF[boots]
    training_images.append(ImageUrlCreateEntry(url=product_img_link,tag_ids=[style_tag_boots.id]))
summary = trainer.create_images_from_urls(project.id, training_images)
print(summary.is_batch_successful, summary.status)

# COMMAND ----------

from itertools import islice
product_img_links = list(islice(slippersDF, 64))
training_images = []
for slippers in list(islice(slippersDF, 64)):
    product_img_link = slippers
    product_img_link = product_img_link.replace("bloomingdales.com", "bloomingdalesassets.com")
    tagList = slippersDF[slippers]
    training_images.append(ImageUrlCreateEntry(url=product_img_link,tag_ids=[style_tag_slippers.id]))
summary = trainer.create_images_from_urls(project.id, training_images)

# COMMAND ----------

from itertools import islice
product_img_links = list(islice(sandalsDF, 64))
training_images = []
for sandals in list(islice(sandalsDF, 64)):
    product_img_link = sandals
    product_img_link = product_img_link.replace("bloomingdales.com", "bloomingdalesassets.com")
    tagList = sandalsDF[sandals]
    training_images.append(ImageUrlCreateEntry(url=product_img_link,tag_ids=[style_tag_sandals.id]))
summary = trainer.create_images_from_urls(project.id, training_images)
print(summary.is_batch_successful)

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