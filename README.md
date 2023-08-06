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
from gorgias_ml.contact_reason import ContactReason
import pandas as pd

# Remove nones
df = pd.read_parquet("data/classification_dataset")
df = df[~df.email_sentence_embeddings.isna()]

# Fit & predict
cr = ContactReason()
cr.fit(df)
preds = cr.predict(df)

# Save
cr.save_artifacts()
cr.save_predictions(preds)
```

From CLI
/!\ Make sure that you removed nones from the data set. You can run the above 2 python code.
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