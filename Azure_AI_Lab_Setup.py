# Databricks notebook source
# MAGIC %md
# MAGIC <code>https://www.azure.com/free</code><br>
# MAGIC <code>https://azure.microsoft.com/en-us/free/free-account-faq/</code>

# COMMAND ----------

# MAGIC %md
# MAGIC In the Azure portal (https://portal.azure.com), select Create a resource > Data + Analytics > Azure Databricks <br>
# MAGIC Under Azure Databricks Service, provide the values to create a Databricks workspace: <br>
# MAGIC Location: East US 2 <br>
# MAGIC Pricing Tier: Trial (Premium)

# COMMAND ----------

# MAGIC %md
# MAGIC Goto https://shell.azure.com and select your Azure Subscription Account <br>
# MAGIC Install Databricks CLI Python package
# MAGIC <code>
# MAGIC   <br>
# MAGIC   pip install databricks-cli --user <br>
# MAGIC   export PATH=$PATH:~/databrickscli/bin <br>
# MAGIC   az account list -o table <br>
# MAGIC   az account set --subscription YOUR_SUBSCRIPTION_ID_HERE
# MAGIC </code>

# COMMAND ----------

# MAGIC %md
# MAGIC Login to https://portal.azure.com <br>
# MAGIC Click on the green "+" sign on the top right to create a resource, search for "Data Science Virtual Machine - Windows 2016"<br>
# MAGIC VM Name: dsvm-workshop<br>
# MAGIC Provide a Login Username and Login Password<br>
# MAGIC Select a subscription that you want to use for this workshop<br>
# MAGIC VM Disk Type: Standard SSD<br>
# MAGIC Resource Group: Create New > DSVMRG<br>
# MAGIC Location: East US2<br>
# MAGIC Size: B4ms (4cores, 16GB RAM, 32GB Local SSD)<br>
# MAGIC Settings: Use Defaults<br>
# MAGIC Accept the Terms and Create VM, takes about 10mins to create the VM<br>
# MAGIC <br>
# MAGIC For more information on the Windows Data Science VM: <br>
# MAGIC https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/provision-vm#tools-installed-on-the-microsoft-data-science-virtual-machine

# COMMAND ----------

# MAGIC %md
# MAGIC From within the Azure Portal, https://portal.azure.com<br>
# MAGIC Search for "dsvm-workshop" from the Search box on the top of the page<br>
# MAGIC Click on "dsvm-workshop   Virtual Machines" search result<br>
# MAGIC Click on "Connect" and Download the RDP file<br>
# MAGIC Double-click the downloaded RDP file and enter the username/password for the VM to access the Remote Desktop<br>
# MAGIC After the VM completes its one-time setup, right-click on the Powershell icon on the taskbar and "Run as Administrator"<br>
# MAGIC Copy the following text and paste it into the Powershell Prompt:<br>
# MAGIC <code>net use Z: \\bigdatacatalog.file.core.windows.net\ai-discovery-data /u:AZURE\bigdatacatalog USE_INSTRUCTOR_PROVIDED_KEY_HERE </code><br>
# MAGIC <br>
# MAGIC Open Windows Explorer to find a Z: drive attached to the VM<br>
# MAGIC Copy the "workshop" folder and paste it into C: drive<br>
# MAGIC You should now have a C:\workshop folder with some files in it<br>