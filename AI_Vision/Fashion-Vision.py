# Databricks notebook source
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

sandalsQueries = ["site:bloomingdales.com Sandals"]
sandalsUrls = bingPhotoSearch(style_tag_sandals.id, sandalsQueries, pages=100)
displayDF(sandalsUrls)

# COMMAND ----------

slippersQueries = ["site:bloomingdales.com Slippers"]
slippersUrls = bingPhotoSearch(style_tag_slippers.id, slippersQueries, pages=100)
displayDF(slippersUrls)

# COMMAND ----------

sneakersQueries = ["site:bloomingdales.com Sneakers"]
sneakersUrls = bingPhotoSearch(style_tag_sneakers.id, sneakersQueries, pages=100)
displayDF(sneakersUrls)

# COMMAND ----------

bootsQueries = ["site:bloomingdales.com Boots"]
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

print(project.id)

# COMMAND ----------

for boots in bootsDF:
    product_img_link = boots
    tagList = bootsDF[boots]
    trainer.create_images_from_urls(project.id, [ImageUrlCreateEntry(url=product_img_link,tag_ids=tagList)])