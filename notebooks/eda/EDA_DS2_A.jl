### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 63a666c8-6c1c-49f5-9dbe-0a9767ec6518
begin
using Pkg
Pkg.activate("../../") 
end

# ╔═╡ 9567bbf2-d953-492d-b3f0-489fab9376e7
# load dependencies and helper functions
begin
	using DataFrames, CSV, Statistics, Plots, StatsPlots, StatsBase,Distributions

	include("../../src/plotting_functions.jl")
	include("../../src/data_clean.jl")
end

# ╔═╡ 6f23dbc8-82a0-11ef-1ea4-7de9c1060469
md"""
# EDA - DS2 (Part A)
## Purpose: 
- Initial look at the data in three other files
- See the state of missing values in the datasets
"""

# ╔═╡ 825f9e29-1b46-4cd1-905b-a7bb6ee6f31c
# Load in files
begin
	
	# # Define the column names
	column_names = [
    	"age", 
    	"sex", 
    	"chest_pain", 
    	"rest_bp", 
    	"serum_chol", 
	    "fasting_blood_sugar", 
		"electrocardiographic", 
	 	"max_heart_rate", 
   	 	"angina", 
    	"oldpeak", 
    	"slope", 
    	"major_vessels", 
    	"thal", 
    	"heart_disease"
	]


	# Open files
	df_clev = CSV.read("../../data/DS2/processed.cleveland.data", 
		header=column_names, DataFrame) 

	df_hung = CSV.read("../../data/DS2/processed.hungarian.data", 
		header=column_names, DataFrame) 

	df_swiss = CSV.read("../../data/DS2/processed.switzerland.data", 
		header=column_names, DataFrame) 

	df_lng_beach = CSV.read("../../data/DS2/processed.va.data", 
		header=column_names,
		DataFrame)

end

# ╔═╡ 9d178d57-3c9f-42d2-8be7-bef72e7ecf63
begin
	function missing_datapoints_percent(df::DataFrame)
		col_size = size(df)[1]
		return [(count(x -> ismissing(x) || x == "?", 
			df[!, col]) / col_size) * 100 for col in names(df)]
	end
end

# ╔═╡ aabd1b26-19b3-4591-bf0c-1585f24ce4cc
md"""
## Quick look at the data
"""

# ╔═╡ cafd9196-b39e-476a-9749-8266a10af465
df_clev

# ╔═╡ 2b0080dc-6320-4c23-b05b-9f6041eef2bf
df_hung

# ╔═╡ 6d4855e6-a013-46e3-8a4c-00805fbae0d5
df_swiss

# ╔═╡ 7fd9e68e-9ce9-4854-a30f-ba5a36341aeb
df_lng_beach

# ╔═╡ 0fbee3ba-de76-4213-b41e-cf8abcdcd62b
md"""
### Comments on quick look
- All three files have at least one column that has the data points as Strings instead of the types they should be such as Int or Float. 
- This is easyly fixed by casting valid Strings to the correct types
- All three files have missing data points either as `missing` type, or as the String "?"
"""

# ╔═╡ 9d6c57b6-e327-4397-a99f-f71196166848
md"""
### How much data is missing?
"""

# ╔═╡ 5a3b798d-e364-4804-9b72-27db24d60bbe
begin

	# Get percentage of missing datapoints per feature
	clev_missing = missing_datapoints_percent(df_clev)

	# Plot the percentage of missing data points per feature
	bar( 
		clev_missing, 
		legend=false, 
		xlabel="Features", 
		ylabel="Percentage Missing", 
		title="Percentage of data missing in Cleveland Dataset", 
		orientation=:vertical, 
		xticks=1:size(clev_missing)[1]
	)


end

# ╔═╡ af000366-be8b-4e93-93bb-5b2e11f2f87e
begin

	# Get percentage of missing datapoints per feature
	missing_hung = missing_datapoints_percent(df_hung)
	
	# Plot the percentage of missing data points per feature
	bar( 
		missing_hung, 
		legend=false, 
		xlabel="Features", 
		ylabel="Percentage Missing", 
		title="Percentage of data missing in Hungarian Dataset", 
		orientation=:vertical, 
		xticks=1:size(missing_hung)[1]
	)
end

# ╔═╡ bf0857f1-7a76-433c-9079-45526d4b49e3
begin

	# Get percentage of missing datapoints per feature
	missing_swiss = missing_datapoints_percent(df_swiss)
	
	# Plot the percentage of missing data points per feature
	bar( 
		missing_swiss, 
		legend=false, 
		xlabel="Features", 
		ylabel="Percentage Missing", 
		title="Percentage of data missing in Swiss Dataset", 
		orientation=:vertical, 
		xticks=1:size(clev_missing)[1]
	)
end

# ╔═╡ 999df9ee-2e33-4552-99b2-dd91db92ec28
begin

	# Get percentage of missing datapoints per feature
	missing_lng_beach = missing_datapoints_percent(df_lng_beach)
	
	# Plot the percentage of missing data points per feature
	bar( 
		missing_lng_beach, 
		legend=false, 
		xlabel="Features", 
		ylabel="Percentage Missing", 
		title="Percentage of data missing in Long Beach Dataset", 
		orientation=:vertical, 
		xticks=1:size(clev_missing)[1]
	)
end

# ╔═╡ b187af11-97e5-4e14-a7b3-9227f18b67ea
println(missing_lng_beach)

# ╔═╡ 58a86246-3109-4190-a62a-e4fc3c955b21
md"""
### Comment on missing data

#### Feature mapping:

| Feature Number | Feature Name           |
|----------------|------------------------|
| 1              | age                    |
| 2              | sex                    |
| 3              | chest_pain             |
| 4              | rest_bp                |
| 5              | serum_chol             |
| 6              | fasting_blood_sugar    |
| 7              | electrocardiographic   |
| 8              | max_heart_rate         |
| 9              | angina                 |
| 10             | oldpeak                |
| 11             | slope                  |
| 12             | major_vessels          |
| 13             | thal                   |
| 14             | heart_disease          |

##### Cleveland
- `major_vessels` has only 1.32% missing
- `thal` has only 0.66% missing
##### Hungarian
Lots more missing data here
- `rest_bp` has 0.34% missing
- `serum_chol` has 7.82% missing
- `fasting_blood_sugar` has 2.72% missing
- `electrocardiographic` has 0.34% missing
- `max_heart_rate` has 0.34% missing
- `angina` has 0.34% missing
- `slope` has 64.63% missing
- `major_vessels` has 98.98% missing
- `thal` has 90.48% missing
#### Swiss 
Lots more missing data here
- `rest_bp` has 1.63% missing
- `fasting_blood_sugar` has 60.98% missing
- `electrocardiographic` has 0.81% missing
- `max_heart_rate` has 0.81% missing
- `angina` has 0.81% missing
- `oldpeak` has 4.88% missing
- `slope` has 13.82% missing
- `major_vessels` has 95.93% missing
- `thal` has 42.28% missing

#### Long Beach
- `rest_bp` has 28.0% missing
- `serum_chol` has 3.5% missing
- `fastingbloodsugar` has 3.5% missing
- `maxheartrate` has 26.5% missing
- `angina` has 26.5% missing
- `oldpeak` has 28.0% missing
- `slope` has 51.0% missing
- `major_vessels` has 99.0% missing
- `thal` has 83.0% missing

---

We can see that both the Hungarian and the Swiss processed datasets have many missing values for certian feature. This is will require imputation. Given that there are so many missing values it is uncertian at this points if some of these features will be usable. 
"""

# ╔═╡ Cell order:
# ╠═6f23dbc8-82a0-11ef-1ea4-7de9c1060469
# ╠═63a666c8-6c1c-49f5-9dbe-0a9767ec6518
# ╠═9567bbf2-d953-492d-b3f0-489fab9376e7
# ╠═825f9e29-1b46-4cd1-905b-a7bb6ee6f31c
# ╠═9d178d57-3c9f-42d2-8be7-bef72e7ecf63
# ╟─aabd1b26-19b3-4591-bf0c-1585f24ce4cc
# ╠═cafd9196-b39e-476a-9749-8266a10af465
# ╠═2b0080dc-6320-4c23-b05b-9f6041eef2bf
# ╠═6d4855e6-a013-46e3-8a4c-00805fbae0d5
# ╠═7fd9e68e-9ce9-4854-a30f-ba5a36341aeb
# ╟─0fbee3ba-de76-4213-b41e-cf8abcdcd62b
# ╟─9d6c57b6-e327-4397-a99f-f71196166848
# ╟─5a3b798d-e364-4804-9b72-27db24d60bbe
# ╟─af000366-be8b-4e93-93bb-5b2e11f2f87e
# ╟─bf0857f1-7a76-433c-9079-45526d4b49e3
# ╠═999df9ee-2e33-4552-99b2-dd91db92ec28
# ╠═b187af11-97e5-4e14-a7b3-9227f18b67ea
# ╟─58a86246-3109-4190-a62a-e4fc3c955b21
