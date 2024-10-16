# Heart Disease - INFO 411 Assignment Two

## Dashboard
- To view go to `http://localhost:1234/edit?id=a1fc6274-8b81-11ef-04cd-f70530cab4ec&isolated_cell_id=9a9ffaf6-43ca-4eda-9b31-7fc658dd4f9a`

## Navigation

- `dashboard.js` is the notebook that creates the dashboard as described above. This answers part **Dashboards (5)** from the assignment spec
- `notebooks/` directory contains...notebooks in different sub-folders
    - `notebooks/eda/` directory contains note books that complete part **Exploratory data analysis (10 marks)** from the assignment spec
    - `notebooks/imputation/` directory contains notebook `imputation.jl`. This contains experiments and analysis of different imputation techniques. It also applies the final technique and saves the files to `data/`. This answers part **Data imputation (5 marks)** from the assignment spec
    - `notebooks/modelling/` contains notebooks that train, analyze and export the various algorithms 
- `src/` directory contains common functions used in various notebooks
- `models/` directory contains exported/saved models that are later used in the `dashboard.jl` and tested on different dataset. 
