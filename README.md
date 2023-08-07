# Contact Reason Prediction

## Project Description

The objective of the project is to predict contact reason for each customer inquery.
## Getting Started

Installation
------------
    $ git clone https://github.com/abdel-ely-ds/classification_project_gorgias.git
    $ cd classification_project_gorgias
    $ pip install -e .
    $ pip install .[api] to run the service
    $ pip install .[gpu] to use faiss gpu
    
Usage
------------

```python
from sklearn.model_selection import train_test_split
from gorgias_ml.contact_reason import ContactReason
import pandas as pd
import gorgias_ml.constants as cst
from sklearn.metrics import precision_recall_fscore_support

# Remove nones & duplicates
df = pd.read_parquet("data/classification_dataset")
df = df[~df.email_sentence_embeddings.isna()]
df = df.drop_duplicates()


x_train, x_val = train_test_split(df, random_state=2023, test_size=0.2)

# Fit & predict
cr = ContactReason()
cr.fit(df)
train_preds = cr.predict(x_train)
val_preds = cr.predict(x_val)

# Eval
train_truth = x_train[cst.TARGET]
val_truth = x_val[cst.TARGET]
precision, recall, f1_score, _ = precision_recall_fscore_support(train_truth, train_preds, average="weighted")

print(
    "-----------Training Scores-------------\n"
    f"Weighted precision: {precision:.2f} \n"
    f"Weighted recall: {recall:.2f} \n"
    f"Weighted f1_score: {f1_score:.2f} \n"
    "-------------------------------------------"
)

precision, recall, f1_score, _ = precision_recall_fscore_support(val_truth, val_preds, average="weighted")

print(
    "-----------Validation Scores-------------\n"
    f"Weighted precision: {precision:.2f} \n"
    f"Weighted recall: {recall:.2f} \n"
    f"Weighted f1_score: {f1_score:.2f} \n"
    "-------------------------------------------"
)

# Save
cr.save_artifacts()
cr.save_predictions(preds)
```

From CLI: 
<br/><h4> /!\ Make sure that you removed nones from the data set. You can run the above 2 python code.</h4>
------------
    $ contact-reason train --data-dir ./data/ --output-dir ./artifacts --df-filename classification_dataset

Run as a service
------------
    $ uvicorn api.app:app

Project Structure
------------

    ├── README.md          <- README of the project.
    ├── data               <- Raw data.
    ├── src                <- source code for training and predicting contact reasons.
    ├── notebooks          <- Jupyter notebooks.
    ├── requirements.txt   <- The requirements file contains all the necessary libs to run the project.
    ├── tests              <- tests forlder.
    ├── api                <- to run the project as a service.
    ├── artifacts          <- saving system artifacts.
    ├── results            <- performance of the system.
    └── noxfile.py         <- black, build, tests.               

--------
