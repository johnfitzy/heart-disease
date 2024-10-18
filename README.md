# Heart Disease - INFO 411 Assignment Two

## NOTE: You will probably require Julia version 1.11.0 to run the the dashboard because we importing out machines from a serialized format. Julia completely broke the way it does serialization in a minor version update (?!)

## Running Project
Run these commands in this order:
```
cd heart-disease
julia --pkgimages=no
import Pkg
Pkg.activate(".")
Pkg.instantiate()
using Pluto;Pluto.run()
```

## Dashboard
- To view go to `http://localhost:1234/edit?id=495cabc2-8cd2-11ef-1f78-fd2accced723&isolated_cell_id=9a9ffaf6-43ca-4eda-9b31-7fc658dd4f9a`

## Navigation

- `dashboard.js` is the notebook that creates the dashboard as described above. This answers part **Dashboards (5)** of the assignment spec
- `report.docx` is the report, this answers part **Report (5)** of the assignment spec
- `notebooks/` directory contains...notebooks, in different sub-folders
    - `notebooks/eda/` directory contains notebooks that complete part **Exploratory data analysis (10 marks)** of the assignment spec
    - `notebooks/imputation/` directory contains notebook `imputation.jl`. This contains experiments and analysis of different imputation techniques. It also applies the final technique and saves the files to `data/`. This answers part **Data imputation (5 marks)** of the assignment spec
    - `notebooks/modelling/` contains notebooks that train, analyze and export the various algorithms. This answers part **Modelling (10 marks)** of the assignment spec
- `data/` directory contains the datasets for the experiments. 
- `src/` directory contains common functions used in various notebooks
- `models/` directory contains exported/saved models that are later used in the `dashboard.jl` and tested on different dataset. 


Key Points:
activate . tells Julia to use the environment in the current directory.
instantiate installs all dependencies listed in the Project.toml and Manifest.toml files, making sure the environment is exactly as intended.
