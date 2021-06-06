# Bertelsmann-Arvato-Capstone-Project
Short instructions of usage

## Needed python libraries
You should have jupyter notebook up and running.
The code was tested with Python 3.7.9.
```bash
$ pip install scikit-learn pandas numpy matplotlib seaborn joblib datetime scipy yellowbrick xlrd tables xgboost imblearn pandas_profiling
```

## File information

| Filename                                    | Description |
|---------------------------------------------|-------------|
| ./project/00_generate-profile-reports.ipynb | Notebook to create profile reports of the datasets. |
| ./project/01_data-exploration-raw.ipynb     | Here Part I - the Data exploration takes place |
| ./project/02_impute_geburtsjahr.ipynb       | The feature GEBURTSJAHR is being reconstructed here | 
| ./project/03_Arvato Project Workbook.ipynb  | This is the main Notebook for Part II (Customer segmentation) and Part III Supervised learning |
| ./project/common_functions.py               | This file provides commonly used python functions |
| ./project/featureengineering.py             | This file contains classes for Data pre-processing and Feature engineering |
| ./proposal.pdf                              | The capstone proposal |
| ./Report.pdf                                | The capstone Report |
