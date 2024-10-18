### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 7c617e8e-3f5b-4a7f-8c38-e4d10ee662d4
begin
using Pkg
Pkg.activate("../../") 
end

# ╔═╡ 8520ae4a-7fc0-11ef-0890-2dfb0d7ebdfe
# load dependencies and helper functions
begin
	using DataFrames, CSV, Statistics, Plots, StatsPlots, StatsBase,Distributions, 	HypothesisTests, Impute


	include("../../src/plotting_functions.jl")
	include("../../src/data_clean.jl")
end

# ╔═╡ 00c5e24d-19a7-4c52-ac2a-829390ca95de
md"""
# EDA - DS2 (Part B)
## Purpose: 
- Brief comparison between `data/DS1/heart.dat` distributions and files in `DS2`
- From `EDA - DS2 Part A` recall there are many missing values, it is assumed you have looked as playbook `eda_ds2-a.jl` before this one. Only features with minimal mising values will be considered here - ie less that 1% missing
- This is not an exhaustive comparrison between each feature and each dataset, therefore we have just picked **three features from each dataset**
- The file `processed.cleveland.data` appears to be super set of `heart.dat`>
"""

# ╔═╡ 2839ea3a-0a95-42fb-99c8-c86a80ac71b8
begin
	# Define the column names
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
	df_ds1 = CSV.read("../../data/DS1/heart.dat", 
		header=column_names, 
		DataFrame) 
	
	df_clev = CSV.read("../../data/DS2/processed.cleveland.data", 
		header=column_names,
		DataFrame) 

	df_hung = CSV.read("../../data/DS2/processed.hungarian.data", 
		header=column_names, 
		DataFrame) 

	# long-beach-va.data

	df_swiss = CSV.read("../../data/DS2/processed.switzerland.data", 
		header=column_names, 
		DataFrame)

	df_lng_beach = CSV.read("../../data/DS2/processed.va.data", 
		header=column_names,
		DataFrame)

end

# ╔═╡ 525a0646-d2c5-4a1e-be73-ac4f8032171a
md"""
# DS1 vs Clevland(DS2)
## Feature Analysis - Sex
"""

# ╔═╡ dccf5d92-d1a1-4f50-a27f-84b393f95323
describe(df_clev.sex)

# ╔═╡ 24803bbf-7d98-4741-a99d-810812a25334
begin
	# Create the two plots

	plot(
		create_binary_bar_plot(
    		data=df_ds1.sex, 
    		title_text="Sex Distribution (DS1)",
    		xlabel_text="Sex",
    		ylabel_text="Count",
    		labels=["Male", "Female"]
		),

		create_binary_bar_plot(
    		data=df_clev.sex, 
    		title_text="Sex Distribution (Cleveland)",
    		xlabel_text="Sex",
    		ylabel_text="Count",
    		labels=["Male", "Female"]
		), layout = @layout([a b]))
end

# ╔═╡ d81d784b-d417-4193-a2a2-6c72db0867e0
md"""
## Feature Analysis: Age
"""

# ╔═╡ 7a319498-f455-4bad-b701-9c7842f2409e
begin
	plot(
		create_hist_plot(df_ds1.serum_chol, 
    	"serum_chol - before", 
        "Density Plot of Serum Cholesterol", 
        "cholesterol", 
    	"Frequency", 
        "Serum Cholesterol", 
        "Density"
	),

	create_hist_plot(df_clev.serum_chol, 
    	"serum_chol - after", 
        "Density Plot of Serum Cholesterol", 
        "cholesterol", 
    	"Frequency", 
        "Serum Cholesterol", 
        "Density"
	),
		layout = @layout([a b])
	)
end

# ╔═╡ 22671832-5d03-4df9-81a3-f0db8fa704ca
println(UnequalVarianceTTest(df_ds1.serum_chol, df_clev.serum_chol))

# ╔═╡ 26fd39f5-feb6-4d8a-8b02-5425639f459e
md"""
### Comments
- We can view from the above plots that df_ds1.age` and `df_clev.age` are bassically the same and `df_ds1.serum_chol` and `df_clev.serum_chol` also have essentially the same probability distribution
- The null hypothesis for `df_ds1.serum_chol` and `df_clev.serum_chol` can't be rejected sugesting there is no statistically significant difference between the datasets. 
- It is likely that `heart.dat(ds1)` is a subset of `processed.cleveland.data`
"""

# ╔═╡ f61d891e-6f86-4019-8c64-a821416010c3
md"""
# DS1 vs Hungarian(DS2)
## Feature Analysis - Resting Blood Pressure  (rest_bp)
"""

# ╔═╡ a4e1287f-a430-4aaa-b9b0-aa4f77690f79
begin

	# Convert Strings to Int
	df_hung.rest_bp = map(x -> clean_value(x, Int), df_hung.rest_bp)
	
	# Convert Union{Missing, Int64} to Int (there is no missing)
	df_hung.rest_bp = coalesce.(df_hung.rest_bp)
	

	plot(
		create_hist_plot(df_ds1.rest_bp, 
    		"rest_bp - DS1", 
        	"Histogram of Resting Blood Pressure", 
	        "rest_bp", 
	    	"Frequency", 
        	"Resting Blood Pressure", 
        	"Density"
		),

		create_hist_plot(collect(skipmissing(df_hung.rest_bp)), 
    		"rest_bp - Hungary", 
        	"Density Plot of Resting Blood Pressure", 
        	"rest_bp", 
    		"Frequency", 
        	"Resting Blood Pressure", 
        	"Density"
		),
		layout = @layout([a b])
	
	)
end

# ╔═╡ 93845f54-1d15-48bb-b9aa-aa51fc4bdba3
println(UnequalVarianceTTest(df_ds1.serum_chol, df_clev.serum_chol))

# ╔═╡ 2a311f86-9404-4989-8214-fbf9ac6ffff1
md"""
## Feature Analysis - Max Heart Rate  (max_heart_rate)
"""

# ╔═╡ 30237426-1e35-4702-9f74-14558eb350b3
begin

	# Convert Strings to Int
	df_hung.max_heart_rate = map(x -> clean_value(x, Int), df_hung.max_heart_rate)

	
	# Drop the 1 missing values and convert to vector of Int
	filtered_df_hung_max_heart_rate = collect(skipmissing(df_hung.max_heart_rate))


	plot(
		create_hist_plot(df_ds1.max_heart_rate, 
    		"max_heart_rate - DS1", 
        	"Histogram of Max Resting Heart Rate", 
	        "max_heart_rate", 
	    	"Frequency", 
        	"Max Resting Heart Rate", 
        	"Density"
		),

		create_hist_plot(filtered_df_hung_max_heart_rate, 
    		"max_heart_rate - Hungary", 
        	"Histogram of Max Resting Heart Rate", 
        	"max_heart_rate", 
    		"Frequency", 
        	"Max Resting Heart Rate", 
        	"Density"
		),
		layout = @layout([a b])
	
	)

end

# ╔═╡ ceaf6dd9-1b81-4da0-b593-9dd2bb8faba6
println(UnequalVarianceTTest(df_ds1.max_heart_rate, filtered_df_hung_max_heart_rate))

# ╔═╡ 1b5ac701-1559-4f3a-b141-859b18bf5739
md"""
### Comments
- Resting Blood Pressure (rest_bp) shows **no significant** statistical difference between `df_ds1.rest_bp` and `df_clev.rest_bp`.
- For `df_ds1.max_heart_rate` and `hung.max_heart_rate`, the null hypothesis **is rejected**, indicating a statistically significant difference between the datasets.
"""

# ╔═╡ 50795050-2a28-4303-9aa0-1120d78415cc
md"""
# DS1 vs Switzerland(DS2)
## Feature Analysis - ST depression induced by exercise relative to rest(oldpeak)
"""

# ╔═╡ e1cbd982-b813-462a-929d-054c905adb2b
begin

		# Convert Strings to Int
	df_swiss.oldpeak = map(x -> clean_value(x, Int), df_swiss.oldpeak)

	
	# Drop the 1 missing values and convert to vector of Int
	filtered_df_swiss_oldpeak = collect(skipmissing(df_swiss.oldpeak))

	
	plot(
		create_hist_plot(df_ds1.oldpeak, 
    		"oldpeak - DS1", 
        	"Histogram of ST depression (oldpeak)", 
	        "oldpeak", 
	    	"Frequency", 
        	"ST depression (oldpeak)", 
        	"Density"
		),

		create_hist_plot(filtered_df_swiss_oldpeak, 
    		"oldpeak - Switzerland", 
        	"Histogram of ST depression (oldpeak)", 
        	"oldpeak", 
    		"Frequency", 
        	"ST depression (oldpeak)", 
        	"Density"
		),
		layout = @layout([a b])
	)
end

# ╔═╡ 61aac936-0e68-46e4-bb85-0ccfe5075787
println(UnequalVarianceTTest(df_ds1.oldpeak, filtered_df_swiss_oldpeak))

# ╔═╡ 655d5e2d-9b0f-4abf-8a7a-2b19716dbb40
begin
	# Convert from String to Int or missing
	df_swiss.angina = map(x -> clean_value(x, Int), df_swiss.angina)
	
	plot(

		create_binary_bar_plot(
			data = df_ds1.angina, 
    		title_text = "Angina - DS1",
    		xlabel_text = "Angina with Exercise",  # X-axis label
    		ylabel_text = "Percentage (%)",        # Y-axis label
    		labels = ["Yes", "No"]  # Labels for the categories 1 (Yes) and 0 (No)
		),


		create_binary_bar_plot(
			data = collect(skipmissing(df_swiss.angina)), # remove missing and to array
    		title_text = "Angina - Swiss",
    		xlabel_text = "Angina with Exercise",  # X-axis label
	    	ylabel_text = "Percentage (%)",        # Y-axis label
    		labels = ["Yes", "No"]  # Labels for the categories 1 (Yes) and 0 (No)
		),
			layout = @layout([a b])
		)
end

# ╔═╡ 3e2121c9-3c72-4c91-ae97-84de34618df8
println(UnequalVarianceTTest(df_ds1.angina, collect(skipmissing(df_swiss.angina))))

# ╔═╡ 13b5f478-dc0c-43d8-af9b-db2c61328c1e
md"""
### Comments
- Oldpeak **is** significantly statistically different between `df_ds1.oldpeak` and `hung.oldpeak` (p < 1e-04)
- Angina **is** significantly statistically different between `df_ds1.angina` and `hung.angina` (p = 0.0357)
"""

# ╔═╡ fb731209-3b3c-4261-8e4c-7ccdc73bc89d
md"""
# DS1 vs Long Beach(DS2)
## Feature Analysis - Resting electrocardiographic results (electrocardiographic)
"""

# ╔═╡ 933e5c6c-3d7c-4b79-9403-fd5cf440b1c8
begin	
	plot(
		create_bar_plot(
    		data=df_ds1.electrocardiographic, 
    		labels=["Normal", "ST-T Abnormality", "LVH"], 
    		values_to_extract=[0, 1, 2], 
    		plot_title="ECG Distribution", 
    		xlabel_text="Electrocardiographic Type", 
    		ylabel_text="Count"
	), 
		create_bar_plot(
    		data=df_lng_beach.electrocardiographic, 
    		labels=["Normal", "ST-T Abnormality", "LVH"], 
    		values_to_extract=[0, 1, 2], 
    		plot_title="ECG Distribution", 
    		xlabel_text="Electrocardiographic Type", 
    		ylabel_text="Count"
	),
		layout = @layout([a b])
	)
end

# ╔═╡ a5df8a8e-3766-4b4a-a3ca-4eb7d9edb206
println(UnequalVarianceTTest(df_ds1.electrocardiographic,df_lng_beach.electrocardiographic))

# ╔═╡ 04ff5df8-7cc7-4146-bb42-f211dddf2b81
md"""
### Comments
- The feature `electrocardiographic` shows a statistically significant difference between `df_ds1.electrocardiographic` and `df_lng_beach.electrocardiographic`, as the null hypothesis **is rejected**.
"""


# ╔═╡ f56b866c-dbbd-42e9-81b5-b9997abf61bb
md"""
## Conclusion
- We can see from this limited compariason between features from DS2 datasets and the DS1 dataset that some features share similar distributions and while others have statistically significant difference between locations. 
"""

# ╔═╡ Cell order:
# ╠═00c5e24d-19a7-4c52-ac2a-829390ca95de
# ╠═7c617e8e-3f5b-4a7f-8c38-e4d10ee662d4
# ╠═8520ae4a-7fc0-11ef-0890-2dfb0d7ebdfe
# ╠═2839ea3a-0a95-42fb-99c8-c86a80ac71b8
# ╠═525a0646-d2c5-4a1e-be73-ac4f8032171a
# ╠═dccf5d92-d1a1-4f50-a27f-84b393f95323
# ╠═24803bbf-7d98-4741-a99d-810812a25334
# ╠═d81d784b-d417-4193-a2a2-6c72db0867e0
# ╠═7a319498-f455-4bad-b701-9c7842f2409e
# ╠═22671832-5d03-4df9-81a3-f0db8fa704ca
# ╟─26fd39f5-feb6-4d8a-8b02-5425639f459e
# ╟─f61d891e-6f86-4019-8c64-a821416010c3
# ╠═a4e1287f-a430-4aaa-b9b0-aa4f77690f79
# ╠═93845f54-1d15-48bb-b9aa-aa51fc4bdba3
# ╟─2a311f86-9404-4989-8214-fbf9ac6ffff1
# ╟─30237426-1e35-4702-9f74-14558eb350b3
# ╟─ceaf6dd9-1b81-4da0-b593-9dd2bb8faba6
# ╟─1b5ac701-1559-4f3a-b141-859b18bf5739
# ╟─50795050-2a28-4303-9aa0-1120d78415cc
# ╟─e1cbd982-b813-462a-929d-054c905adb2b
# ╠═61aac936-0e68-46e4-bb85-0ccfe5075787
# ╟─655d5e2d-9b0f-4abf-8a7a-2b19716dbb40
# ╠═3e2121c9-3c72-4c91-ae97-84de34618df8
# ╟─13b5f478-dc0c-43d8-af9b-db2c61328c1e
# ╠═fb731209-3b3c-4261-8e4c-7ccdc73bc89d
# ╠═933e5c6c-3d7c-4b79-9403-fd5cf440b1c8
# ╠═a5df8a8e-3766-4b4a-a3ca-4eb7d9edb206
# ╟─04ff5df8-7cc7-4146-bb42-f211dddf2b81
# ╟─f56b866c-dbbd-42e9-81b5-b9997abf61bb
