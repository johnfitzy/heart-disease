### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ e5023cfe-9f75-4cda-996e-3af36e9ea599
begin
using Pkg
Pkg.activate("../../") 
end

# ╔═╡ 92a554a2-ff53-477b-8ccf-4a178ee01672
using DataFrames, CSV, Statistics, Plots, StatsPlots, StatsBase,Distributions, MultivariateStats, Random, MLUtils, HypothesisTests

# ╔═╡ f1b4cd6a-68be-4fb7-a4ee-47b7fb5f68ca
include("../../src/plotting_functions.jl")

# ╔═╡ f298fb92-9015-4de9-942e-11206ae80183
md"""
| Variable Name           | Role    | Type         | Description                                                      |
|-------------------------|---------|--------------|------------------------------------------------------------------|
| age                     | Feature | Continuous   | Age                                                              |
| sex                     | Feature | Binary       | Sex                                                              |
| chest-pain              | Feature | Categorical  | Chest pain type                                                  |
| rest-bp                 | Feature | Continuous   | Resting blood pressure                                           |
| serum-chol              | Feature | Continuous   | Serum cholesterol (mg/dl)                                        |
| fasting-blood-sugar     | Feature | Binary       | Fasting blood sugar > 120 mg/dl                                  |
| electrocardiographic    | Feature | Categorical  | Resting electrocardiographic results                             |
| max-heart-rate          | Feature | Continuous   | Maximum heart rate achieved                                      |
| angina                  | Feature | Binary       | Exercise induced angina                                          |
| oldpeak                 | Feature | Continuous   | ST depression induced by exercise relative to rest               |
| slope                   | Feature | Integer      | The slope of the peak exercise ST segment                        |
| major-vessels           | Feature | Continuous   | Number of major vessels (0-3) colored by fluoroscopy             |
| thal                    | Feature | Categorical  | Thal: 3 = normal; 6 = fixed defect; 7 = reversible defect        |
| heart-disease           | Target  | Integer      | Diagnosis of heart disease                                       |

"""

# ╔═╡ a174259c-5822-4aa1-b162-d07deef886af
begin
	# Open file as CSV, heart.csv has heart_disease mapped from 1 and 2 to 0 and 1
	# otherwise it is no different to heart.data
	df = CSV.read("../../data/DS1/heart.csv",
		DataFrame) 
end

# ╔═╡ d6c8603b-569f-4c02-a3fb-3d0e74bbf638
describe(df)

# ╔═╡ e804fdc5-cca9-4912-8c51-d39accde1d3f
md"""
## Part One Ideas

NOTE: just hear as some ideas delete later

1. Understand the Data Distribution

- Descriptive Statistics: Calculate summary statistics (mean, median, standard deviation, min, max, etc.) for each attribute to understand the range and central tendencies.

- Distribution Plots: Use histograms or density plots for continuous variables (e.g., age, rest-bp, serum-chol, max-heart-rate, oldpeak) to visualize their distributions.

- Box Plots: Create box plots to identify outliers in continuous variables. This can help you detect anomalies or extreme values.

"""

# ╔═╡ 35fa6501-bd25-44ef-baca-f5bf27d55820
md"""
# Part One: Exploratory Data Analysis
"""

# ╔═╡ 24dc537e-6786-40e2-9305-a21dbe11d305
md"""

## Data Sneak Peak
| age  | sex  | chest_pain | rest_bp | serum_chol | fasting_blood_sugar | electrocardiographic | max_heart_rate | angina | oldpeak | slope | major_vessels | thal | heart_disease |
|------|------|------------|---------|------------|---------------------|----------------------|----------------|--------|---------|-------|---------------|------|---------------|
| 70.0 | 1.0  | 4.0        | 130.0   | 322.0      | 0.0                 | 2.0                  | 109.0          | 0.0    | 2.4     | 2.0   | 3.0           | 3.0  | 2             |
| 67.0 | 0.0  | 3.0        | 115.0   | 564.0      | 0.0                 | 2.0                  | 160.0          | 0.0    | 1.6     | 2.0   | 0.0           | 7.0  | 1             |
| 57.0 | 1.0  | 2.0        | 124.0   | 261.0      | 0.0                 | 0.0                  | 141.0          | 0.0    | 0.3     | 1.0   | 0.0           | 7.0  | 2             |

From the above we can see some initial properties of the data
1) Features are Categorical, Continuous and Binary types
2) Many of the features are in different scales


"""

# ╔═╡ f834790d-2ff3-42a7-9570-2bdf8526d0d4
md"""
### Feature Analysis: Sex
"""

# ╔═╡ 1847271c-13d1-40e9-8510-5df10c8d3387
md"""
###### Comments:
- 67.8 % are male
- 32.2 % are female
"""

# ╔═╡ e8ae2135-8cbd-4541-9736-7b9f4d89c0d5
begin
	create_binary_bar_plot(
		data=df.sex, 
		title_text="Sex Distribution (Percentage)",
		xlabel_text="Sex",
		ylabel_text="Count",
		labels=["Male", "Female"]		
	)
end

# ╔═╡ 69956da7-2ec9-446f-b820-9918cb7f7a2a
md"""
### Feature Analysis: Age
"""

# ╔═╡ ea135519-5c1c-4133-96af-4217ce2e8125
describe(df.age)

# ╔═╡ 11cb6c44-506c-4261-a07d-27a6bdd3a7ea
md"""

##### Comments on Age feature:
- The age data closely follows a normal distribution, as shown by the QQ Plot, although there are some outliers at the tails, indicating deviations at the extremes of the age spectrum.
- The violin plot suggests a slight negative skew, with a tendency toward older individuals, implying that ages above the mean are slightly more prevalent.
- A noticeable bump around the 40-45 year range suggests a local concentration of data points, which may reflect a particular subgroup or trend within the data.
- The standard deviation of 9.109067 is about 16.7% of the mean (54.433333), indicating moderate variability. This suggests that while there is some spread in the ages, most data points remain relatively close to the mean, reflecting a reasonably consistent age distribution with moderate dispersion.
- The proximity of the median (55) to the mean further supports the observation of a near-symmetrical distribution, despite the slight skew and presence of outliers.

"""

# ╔═╡ 004da6e2-b821-4096-b2e9-5f2ab5dc079e
begin
	create_hist_density_plot(df.age, 
    	"Age Distribution", 
        "Density Plot of Age", 
        "Age", 
    	"Frequency", 
        "Age", 
        "Density"
	)
end

# ╔═╡ d374d9b1-dcad-44a2-9020-ddeb669e435c
begin
	# Plot violin dot and box plot for age
	create_combined_plot(df.age, "Age Violin, Dot and Box Plot", "Frequency", "Age")

end

# ╔═╡ 90c1db0d-67a9-4dff-b5dd-bd9b23339f0e
begin
	# Plot QQ Plot to inspect distribution agains normal distribution
	create_qqplot(df.age, "QQ Plot of Age")
end

# ╔═╡ 95398b93-4809-4339-afa8-d653dd1b52db
md"""
### Feature Analysis: Chest Pain
"""

# ╔═╡ ce90bd00-69a2-4d7e-a96d-875a6eac6b7d
describe(df.chest_pain)

# ╔═╡ b280a319-cfa7-4b79-8a7e-7805d9dcf808
md"""
##### Comments on Chest Paint feature:
- Typical Angina 7.4%
- Atypical Angina 15.6%
- Non-Anginal Pain 29.3%
- Asymptomatic 47.8%
"""

# ╔═╡ 126128d5-8690-4cf0-abab-0a58229ef111
begin
	create_bar_plot(
    	data=df.chest_pain, 
    	labels=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    	values_to_extract=[1, 2, 3, 4], 
    	plot_title="Chest Pain Distribution", 
    	xlabel_text="Chest Pain Type", 
    	ylabel_text="Count"
	)
end

# ╔═╡ 16748bd6-6f85-48cf-8043-c3ab2c8abbb9
md"""
## Feature Analysis: Resting Blood Presure	
"""

# ╔═╡ ffe1d39f-37b7-440e-befd-b31cc696fee3
describe(df.rest_bp	)

# ╔═╡ 06b8a523-96c1-48a5-88bb-1b5a112e92fd
md"""
##### Comments on Resting Blood Pressure feature:
- The probability distribution is somewhat close to a normal distribution but exhibits a slight positive skew, with lower blood pressure readings being more frequent and a longer right tail indicating higher values.
- The data contains several outliers, with some very high readings indicating elevated blood pressure and a few low outliers. These extremes suggest the presence of individuals with unusual resting blood pressure levels, both high and low.
- The standard deviation of 17.861608 is about 13.6% of the mean (131.344444), indicating moderate variability relative to the mean. This suggests that the data points are relatively close to the mean, reflecting a reasonably tight distribution compared to the overall scale of blood pressure values.
- The median (130) is close to the mean, which indicates a reasonably symmetric central tendency, despite the presence of skew and outliers.
- The interquartile range (120 to 140) captures the middle 50% of the data, showing a typical range of resting blood pressure values that are relatively close to the average, supporting the observation of moderate variability.
"""

# ╔═╡ face61a1-a2d3-4700-a886-184cfb19700a
begin

	create_hist_density_plot(df.rest_bp, 
		"Blood Pressure Distribution",  # Title for the histogram
        "Density Plot of Blood Pressure",  # Title for the density plot
        "Blood Pressure",  # X-axis label for the histogram
    	"Frequency",       # Y-axis label for the histogram
        "Blood Pressure",  # X-axis label for the density plot
        "Density")         # Y-axis label for the density plot

end

# ╔═╡ fd88d57c-7c1b-49ff-9d0a-bbafbf1289b9
begin

	create_combined_plot(df.rest_bp, 
		"Resting Blood Presure Violin, Dot and Box Plot", 
		"Frequency", 
		"Blood Presure")

end

# ╔═╡ 718c6ba5-9c1f-4dfd-8e3c-fc2dd58f9fb8
begin
	# QQ Plot to inspect distribution agains normal distribution
	create_qqplot(df.rest_bp, "QQ Plot of Resting Blood Pressure")
end

# ╔═╡ 17fe2f69-2610-431f-af05-434e5ec1e6ef
md"""
### Feature Analysis: Serum Cholesterol
"""

# ╔═╡ 7d50fa1c-77b2-4a48-b825-1c3664684f42
describe(df.serum_chol)

# ╔═╡ b22b3bb3-ea6b-41e5-bd74-62f5145c4416
md"""
#### Comments on Serum Cholesterol feature:
- The probability distribution is somewhat close to a normal distribution; however, the presence of high cholesterol outliers creates a significant positive skew, resulting in a long right tail. This suggests that while most cholesterol levels are centered around the mean, there are individuals with unusually high levels.
- The standard deviation of 51.686237 is approximately 20.7% of the mean (249.659259), indicating moderate variability in relation to the average cholesterol level. This suggests that the data points are fairly spread around the mean, reflecting moderate dispersion compared to the typical value.
- The median (245) is slightly lower than the mean, reinforcing the presence of skew toward higher values due to the outliers.
- The interquartile range (213 to 280) shows where the central 50% of the data lies, which is relatively close to the mean but skewed upward, highlighting that while typical values are consistent, extreme values contribute significantly to the overall spread.
- Overall, the data reflects a distribution where most individuals have cholesterol levels close to the mean, but the influence of high outliers pulls the average up, leading to the observed skew and extended tail.
"""

# ╔═╡ 9113278a-604c-44fc-a51a-610c4ce37064
	create_hist_density_plot(df.serum_chol, 
    	"Serum Cholesterol Distribution", 
        "Density Plot of Serum Cholesterol", 
        "cholesterol", 
    	"Frequency", 
        "Serum Cholesterol", 
        "Density"
	)

# ╔═╡ aaf1e4b6-72bd-4385-a79d-ce8759f1d90a
begin

	create_combined_plot(df.serum_chol, 
		"Serum Cholesterol Violin, Dot and Box Plot", 
		"Frequency", 
		"Serum Cholesterol")

end

# ╔═╡ f0cd7ffb-4f3e-4e3a-a6cb-4f3582b9094f
	# QQ Plot to inspect distribution agains normal distribution
	create_qqplot(df.serum_chol, "QQ Plot of Serum Cholesterol")

# ╔═╡ d7986e82-1825-4abc-9a0e-0f2fb9ed80fb
md"""
### Feature Analysis: Fasting Blood Sugar > 120 mg/dL 
"""

# ╔═╡ 76b2416d-fd48-4189-842b-157ebea1ed97
md"""
##### Comments on Fasting Blood Sugar feature:
- 14.8% of people had elevated fasting blood sugar levels
- 85.2% of people had normal fasting blood sugar levels
"""

# ╔═╡ 84a429d3-653b-4572-8458-b534dd415899
begin
	create_binary_bar_plot(
    	data = df.fasting_blood_sugar, 
    	title_text = "Fasting Blood Sugar Distribution (Percentage)",  # Title of the plot
    	xlabel_text = "Fasting Blood Sugar",  # X-axis label
    	ylabel_text = "Count",       # Y-axis label
    	labels = ["Elevated", "Normal"]  # Labels for the binary categories
	)
end

# ╔═╡ c84c8d42-152d-4e6e-bf06-099774678e50
md"""
### Feature Analysis: Resting Electrocardiographic
"""

# ╔═╡ 6760f6ec-0f99-4ef6-9752-3bcb46150955
begin
	create_bar_plot(
    	data=df.electrocardiographic, 
    	labels=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], 
    	values_to_extract=[0, 1, 2], 
    	plot_title="Resting Electrocardiographic Results Distribution", 
    	xlabel_text="Electrocardiographic Type", 
    	ylabel_text="Count"
	)
end

# ╔═╡ 32fe8106-97ed-4a48-8980-e221a0ea79fb
md"""
### Feature Analysis: Angina with Exercise (exang)
"""

# ╔═╡ 9d893067-18be-4982-9b42-f18de0a0ed5f
md"""
##### Comments on Angina with Exercise feature:
- 33.0% have exercise induced angina
- 77.0% do not have exercise induced angina
"""

# ╔═╡ 26dde9f9-ad73-454c-b475-dfc09894f400
# Function call to create a percentage bar plot for df.angina
create_binary_bar_plot(
	data = df.angina, 
    title_text = "Angina with Exercise Distribution (Percentage)", #Title of the plot
    xlabel_text = "Angina with Exercise",  # X-axis label
    ylabel_text = "Percentage (%)",        # Y-axis label
    labels = ["Yes", "No"]  # Labels for the categories 1 (Yes) and 0 (No)
)

# ╔═╡ 53503b2d-e4c7-4570-8262-20e8ffb50156
md"""
### Feature Analysis: Max Heart Rate
"""

# ╔═╡ aa772ca5-cedd-4805-add7-0404048b6e2e
describe(df.max_heart_rate)

# ╔═╡ bfc09914-c84d-4e70-8ef3-57e201c74b96
md"""
##### Comments on Max Heart Rate feature:
- The data exhibits a negative skew with a long left tail, indicating that there are outliers with unusually low maximum heart rates. This skew suggests that while most individuals achieve higher maximum heart rates, some have significantly lower values that pull the distribution leftward.
- The standard deviation is 23.165717, which is approximately 15.5% of the mean (149.677778). This indicates moderate variability relative to the average, suggesting that the heart rate values are spread moderately around the mean, reflecting a fairly consistent range for most individuals.
- The median (153.5) is slightly higher than the mean, further supporting the negative skew, as more data points lie above the average, with fewer, but more extreme, lower values.
- The interquartile range (133 to 166) captures the central 50% of the data, indicating that most maximum heart rates are relatively close to the mean, despite the presence of lower outliers.
- Overall, the data suggests that while most maximum heart rates are clustered around typical values, the left skew highlights the influence of individuals with notably lower heart rates, contributing to the overall spread and shape of the distribution.

"""

# ╔═╡ 7eb478b5-6103-44af-bd28-48e5f2fd9080
# Histogram and Density Plot for max_heart_rate
create_hist_density_plot(df.max_heart_rate, 
                         "Max Heart Rate Distribution", 
                         "Density Plot of Max Heart Rate", 
                         "Max Heart Rate", 
                         "Frequency", 
                         "Max Heart Rate", 
                         "Density")

# ╔═╡ 26f71e4e-34df-44a6-89e3-e2e40db03cf7
# Combined Violin, Dot, and Box Plot for max_heart_rate
create_combined_plot(df.max_heart_rate, 
                     "Max Heart Rate Violin, Dot, and Box Plot", 
                     "Frequency", 
                     "Max Heart Rate")

# ╔═╡ bc4edeec-229e-48e5-b29a-a6bafb8040c2
# QQ Plot for max_heart_rate
create_qqplot(df.max_heart_rate, "QQ Plot of Max Heart Rate")

# ╔═╡ 579c4514-ae8e-4321-9aef-6977f2dacba2
md"""
### Feature Analysis: Oldpeak
"""

# ╔═╡ 80f1f0c8-d4b8-4d49-a532-d73a477046c7
describe(df.oldpeak)

# ╔═╡ 5dd78dbc-5e34-4c03-bd9a-f6c1de673106
md"""
##### Comments on Oldspeak feature:
- The data is heavily positively skewed, with a long right tail, indicating that most values are concentrated at lower levels of ST depression, while a few cases exhibit significantly higher values, extending the distribution to the right.
- The standard deviation is 1.145210, which is approximately 109% of the mean (1.050000), indicating high variability relative to the average value. This large spread suggests that while many individuals have low ST depression levels, there is considerable dispersion with some experiencing much higher values.
- The median (0.8) is below the mean, highlighting the positive skew and indicating that over half of the observations fall below the average value, with outliers on the higher end pulling the mean upward.
- The interquartile range (0.0 to 1.6) suggests that the central 50% of the data is relatively concentrated at the lower end, emphasizing that most individuals experience minimal to moderate levels of ST depression.
- Overall, the distribution reflects a pattern where lower values are most common, but the presence of high outliers creates a long tail, driving the overall skew and variability observed in the data.

"""

# ╔═╡ a4701191-219f-4c3f-97f4-8dc4b35dd410
# Histogram and Density Plot for oldspeak
create_hist_density_plot(df.oldpeak, 
                         "Oldspeak Distribution", 
                         "Density Plot of Oldspeak", 
                         "Oldspeak", 
                         "Frequency", 
                         "Oldspeak", 
                         "Density"
						)

# ╔═╡ a407201e-5260-49ea-b48f-376b1714169b
# Combined Violin, Dot, and Box Plot for oldpeak
create_combined_plot(df.oldpeak, 
                     "Oldpeak Violin, Dot, and Box Plot", 
                     "Frequency", 
                     "Oldpeak")

# ╔═╡ 67400cd2-9892-4502-a5de-19fdecd9598a
# QQ Plot for oldpeak
create_qqplot(df.oldpeak, "QQ Plot of Oldpeak")

# ╔═╡ 027d26f8-5177-4be1-b333-a7810792c6e7
md"""
### Feature Analysis: Slope
"""

# ╔═╡ d6c1628b-d144-4cd1-95d1-9ae7dd7e5d39
describe(df.slope)

# ╔═╡ 9d3c1583-eac0-4d38-9a86-d7db17865b6f
md"""
##### Comments on Slope feature:
- Upsloping 7.4%
- Flat 15.6%
- Downsloping 29.3%
"""

# ╔═╡ 4d50430f-1fa0-4eee-8cf1-ef81e56f7402
begin
	create_bar_plot(
    	data=df.slope, 
    	labels=["Upsloping", "Flat", "Downsloping"], 
    	values_to_extract=[1, 2, 3], 
    	plot_title="Slope of Peak Exercise ST Segment Distribution", 
    	xlabel_text="Slope Type", 
    	ylabel_text="Count"
	)
end

# ╔═╡ 064e0ac4-7458-4a19-8e12-0395eb92a654
md"""
### Feature Analysis: Major Vessels
"""

# ╔═╡ 704e6473-b07d-45a9-bf72-394da0ccdefe
df.major_vessels

# ╔═╡ c5056e29-9545-4a94-bc65-6d9fd562424e
describe(df.major_vessels)

# ╔═╡ f044966e-132e-4539-acc7-09e009b9ad4d
md"""
##### Comments on Major Vessesl feature:
- The data is heavily positively skewed with a long right tail, indicating that the majority of observations have zero major vessels colored by fluoroscopy, while a smaller number of cases have high-er values.
- The standard deviation is 0.943896, which is approximately 141% of the mean (0.670370). This high relative variability suggests a wide spread compared to the average value, primarily driven by the presence of higher values and the skewed nature of the data.
- The median (0.000000) and the first quartile (0.000000) indicate that at least half of the data points have no major vessels colored, underscoring the concentration of data at the lower end.
- The third quartile (1.000000) and maximum (3.000000) highlight the extent of variability, with a smaller proportion of individuals having more vessels involved, which significantly contributes to the long right tail.
- Overall, the distribution reflects a dataset where the majority of cases have no vessels colored, but a minority with higher counts creates a substantial skew, pulling the mean up and increasing overall variability.


"""

# ╔═╡ 6cf11a9a-a837-457d-8649-afbdf8926304
# Histogram and Density Plot for major_vessels
create_hist_density_plot(df.major_vessels, 
                         "Major Vessels Distribution", 
                         "Density Plot of Major Vessels", 
                         "Major Vessels", 
                         "Frequency", 
                         "Major Vessels", 
                         "Density", 4) # bins 5

# ╔═╡ 34809859-74e2-410b-b623-6212def3cc53
# Combined Violin, Dot, and Box Plot for major_vessels
create_combined_plot(df.major_vessels, 
                     "Major Vessels Violin, Dot, and Box Plot", 
                     "Frequency", 
                     "Major Vessels")

# ╔═╡ 591b6955-22d5-4b30-9cc6-c2fa57adbc92
# QQ Plot for major_vessels
create_qqplot(df.major_vessels, "QQ Plot of Major Vessels")

# ╔═╡ ea52f75b-8893-41c1-b53b-0b706dc1f56a
md"""
### Feature Analysis: Thal
"""

# ╔═╡ a060652c-9af9-4377-acd5-11b357529aca
md"""
##### Comments on Thal feature:

- Normal 7.4%
- Fixed Defect 15.6%
- Reversible Defect 29.3%

"""

# ╔═╡ b908bc5c-c63d-4f00-9471-28eb51cc59fa
begin
	create_bar_plot(
    	data=df.thal, 
    	labels=["Normal", "Fixed Defect", "Reversible Defect"], 
    	values_to_extract=[1, 2, 3], 
    	plot_title="Thal Categories", 
    	xlabel_text="Thal Type", 
    	ylabel_text="Count"
	)
end

# ╔═╡ 7518ed7e-9f45-4cdc-b1d4-6fa6cf30b609
md"""
## Dimension Reduction: PCA
"""

# ╔═╡ 75ec9e54-15ae-40f1-b249-caccc583af99
begin
	# Keep the 14th colum|n as the labels (binary: 0 or 1 for heart disease)
	labels = df[:, 14]
	
	X = transpose(Matrix(df[:, 1:13]))
	
	X1 = standardize(ZScoreTransform, X, dims=2)
end

# ╔═╡ 601d2031-8b6f-481a-8e2a-e5a3af97878a
M = fit(PCA, X1; maxoutdim=2)	

# ╔═╡ fdf5f45c-43db-4813-adac-242b554fe9af
begin
	# Predict to get outputs from X1_test
	Y = predict(M, X1)
	
	no_disease = Y[:, labels .== 0]
	has_disease = Y[:, labels .== 1]

	# Plot points from reduced dims with no disease
	scatter(no_disease[1, :], 
		no_disease[2, :], 
    	marker=:circle, 
		color=:blue, 
		label="No Disease", 
    	xlabel="PC1", 
		ylabel="PC2", 
		zlabel="PC3"
	)
	
	# Plot points from reduced dims with disease
	scatter!(has_disease[1, :], 
		has_disease[2, :], 
    	marker=:circle, 
		color=:red,
		label="Has Disease"
	)
	
end

# ╔═╡ 11f9c8f5-e0ef-4dc0-82b0-17e20ecc88c5
md"""
##### Comments on PCA:

###### General

- There appears to be a loose separation of points with and without heart disease which may indicate an inherate differenece between the two classes in terms of features. 
- The loose sepeartion seems to be mostly along PC1 and probably indicates the features with the highest loadings in PC1 are resposible for the majority of the difference between the two classes
- The dataset started with 13 features/dimentions and the PCA reduced it to 2 PCA's
- A principal ratio of ~0.3665 indicates that the first two principal components combined explain ~36.65% of the total variance in the original data.

###### Loadings

These are often the top 3 contributors to the variance captured in the 2 PCA's

**PC1 (First Principal Component):**
- Oldpeak: Highest positive loading, indicating a strong influence on PC1.
- Max Heart Rate: Highest negative loading, meaning it contributes significantly but in the opposite direction.
- Slope: Another strong positive influence on PC1.
**PC2 (Second Principal Component):**
- Sex: Highest negative loading, showing a strong inverse relationship with PC2.
- Serum Cholesterol: Highest positive loading, indicating a significant contribution to PC2.
- Age: Also strongly contributes to PC2 in a positive direction.
"""

# ╔═╡ a7b566ab-16f6-4ae4-8340-3e6d3a155215
md"""
# Correlations:
- Explore the correlations of each attribute against the presence of heart disease

-> Correlation Matrix

-> Biserial Correlation Test

-> Chi-Square Test
"""

# ╔═╡ 2f2b1b7f-05a0-449a-bc29-0d2204fdd2f6
md"""
# Correlation Matrix
"""

# ╔═╡ 1d746ebb-03db-4bfb-a10b-ffc0a2a3b124
begin
var_names= names(df)	
correlation_matrix = cor(Matrix(df[:, 1:end]))
heatmap(correlation_matrix,title="Correlation Matrix", size= (900, 750), xtickfontsize=6, xlabel="Attributes", xticks=(1:14, var_names), xrot=45, yticks=1:14, label="Attributes")
end

# ╔═╡ 97587d11-3cab-47d8-bd6f-779195da40a5
md""" Based on the correlation heatmap, there is a strong negative correlation effect of attribute number 8= maximum heart rate to those attributes after 8. 

- note that those with maximum HR achieved, less likely to have heart disease, less likely to have angina during exercise, less likely to have narrowed vessels.
- age/sex/chest/angina history is positively correlated with vessel related attributes.
- also noted the upper right quadrant are more positively correlated with each other such as angina during exercise, ST depression at exercise, downsloping ST, more than 1 narrowed vessels, thalaesemia value. These seem to correlate being diagnosed with heart disease"""

# ╔═╡ 284be63b-61e9-45f9-8bde-0a654107d0eb
md"""
# Biserial Correlation Tests
- This is to compare correlation between continuous variables against binary target (heart_disease)
"""

# ╔═╡ cb332288-3b87-4dd7-8041-d1caea555879
continuous_var = ["age", "rest_bp", "serum_chol", "max_heart_rate", "oldpeak"]

# ╔═╡ ec4cc343-7bbc-4824-ab1f-9d19629331ed
categorical_var = select(df, Not(continuous_var))

# ╔═╡ be317312-cf38-4171-b2da-3d1282179088
begin
biserial_data=[]	
function biserial_correlation(continuous_var, binary_var)
    M1 = mean(continuous_var[binary_var .== 1])
    M0 = mean(continuous_var[binary_var .== 0])
    S = std(continuous_var)
    p = mean(binary_var)
    q = 1 - p
    φ = pdf(Normal(), quantile(Normal(), p))
    r_pb = (M1 - M0) / S * sqrt(p * q) / φ

	r_pb = round(r_pb, digits=4)
	
	return r_pb
end
# Loop through each continuous variable and calculate biserial correlation
for var in continuous_var
    continuous_data = df[!, var]
    binary_data = df.heart_disease
    correlation = biserial_correlation(continuous_data, binary_data)
    push!(biserial_data, correlation)
	println("Biserial Correlation between heartdis and $var: ", correlation)
	
end
end

# ╔═╡ dbc582ea-d72c-498b-8cb4-0d19b6d87458
begin
bar_chart = bar(biserial_data, xticks= (1:5, continuous_var), titlefontsize=10, title= "Biserial Correlation between Heart Disease and Continuous Attributes", legend = :none)
end

# ╔═╡ 36b9e359-c861-43fc-9a87-19e30afd5b24
md"""
Based on the above findings on biserial correlation calculations of continuous attributes:

- There is a moderate positive correlation for age with heart disease diagnosis (i.e., with increasing age, there is increase risk of diagnosis of heart disease)

- There is a mild relative positive correlation for cholesterol levels with heart disease diagnosis (i.e., with increasing cholesterol levels, there is increase risk of diagnosis of heart disease)

- There is a strong positive correlation for oldpeak (this represents ST depression induced by exercise) with heart disease diagnosis (i.e., with increasing cholesterol levels, there is increase risk of diagnosis of heart disease)

- There is a strong negative correlation for maxhr (this represents max) with heart disease diagnosis (i.e., with increasing max heart rate, there is decrease risk of diagnosis of heart disease)
	"""

# ╔═╡ 9f9c29a9-5213-4547-8c67-be914a39be15
begin
categorical_col = [:sex, :chest_pain,:fasting_blood_sugar, :electrocardiographic	,:angina, :slope,:major_vessels,:thal]
hd_col= :heart_disease
chqtest = DataFrame(variable=String[], statistic=Float64[], pvalue=Float64[])
for col in categorical_col
    # Create a contingency table
    contingency_table = combine(groupby(df, [col, hd_col]), nrow => :count)
    
    # Convert the contingency table to a matrix
    observed = DataFrames.unstack(contingency_table, col, hd_col, :count) |> Matrix

	observed = convert(Matrix{Int}, observed)
    # Perform the Chi-Square test
    chi_square_test = ChisqTest(observed)
	cqt_val= round(chi_square_test.stat, digits=3)
	pvaluect= round(pvalue(chi_square_test), sigdigits=3)
    
    # Print the results with p-values
	push!(chqtest, (string(col), cqt_val, pvaluect))
    println("Chi-Square test for $col:")
    println("Test Statistic: ", cqt_val)
	println("P-value:", pvaluect)
    println()
end
end

# ╔═╡ bd12b86d-01fe-454a-b1af-05b3bc7f8754
chqtest

# ╔═╡ cb7d9e99-c183-42b0-a345-69d4a89416e7
bar(chqtest.variable, chqtest.statistic, legend=false, title="Chi-Square Test Statistics", xlabel="Variable", xrot=45, ylabel="Test Statistic")

# ╔═╡ 3baed477-300c-491f-b88c-64427f8f42a6
md"""
The above is a Chi-square analysis of all the categorical data against the heart disease target. Based on the findings, all except fasting blood sugar levels are statiscally significant. In other words, sex (i.e. male), history of chest pain, abnormal resting ecg, angina during exercise, downsloping of ST segment, increase in narrowing vessels, higher thal value are contributing risk factors for presence of heart disease.
- As part of the EDA, these categorical features could be used as features in developing predictive modelling for diagnosis for heart disease.
"""

# ╔═╡ Cell order:
# ╠═e5023cfe-9f75-4cda-996e-3af36e9ea599
# ╠═92a554a2-ff53-477b-8ccf-4a178ee01672
# ╠═f1b4cd6a-68be-4fb7-a4ee-47b7fb5f68ca
# ╠═f298fb92-9015-4de9-942e-11206ae80183
# ╠═a174259c-5822-4aa1-b162-d07deef886af
# ╠═d6c8603b-569f-4c02-a3fb-3d0e74bbf638
# ╟─e804fdc5-cca9-4912-8c51-d39accde1d3f
# ╟─35fa6501-bd25-44ef-baca-f5bf27d55820
# ╟─24dc537e-6786-40e2-9305-a21dbe11d305
# ╟─f834790d-2ff3-42a7-9570-2bdf8526d0d4
# ╟─1847271c-13d1-40e9-8510-5df10c8d3387
# ╠═e8ae2135-8cbd-4541-9736-7b9f4d89c0d5
# ╠═69956da7-2ec9-446f-b820-9918cb7f7a2a
# ╠═ea135519-5c1c-4133-96af-4217ce2e8125
# ╟─11cb6c44-506c-4261-a07d-27a6bdd3a7ea
# ╠═004da6e2-b821-4096-b2e9-5f2ab5dc079e
# ╠═d374d9b1-dcad-44a2-9020-ddeb669e435c
# ╠═90c1db0d-67a9-4dff-b5dd-bd9b23339f0e
# ╠═95398b93-4809-4339-afa8-d653dd1b52db
# ╠═ce90bd00-69a2-4d7e-a96d-875a6eac6b7d
# ╠═b280a319-cfa7-4b79-8a7e-7805d9dcf808
# ╠═126128d5-8690-4cf0-abab-0a58229ef111
# ╠═16748bd6-6f85-48cf-8043-c3ab2c8abbb9
# ╠═ffe1d39f-37b7-440e-befd-b31cc696fee3
# ╟─06b8a523-96c1-48a5-88bb-1b5a112e92fd
# ╠═face61a1-a2d3-4700-a886-184cfb19700a
# ╠═fd88d57c-7c1b-49ff-9d0a-bbafbf1289b9
# ╠═718c6ba5-9c1f-4dfd-8e3c-fc2dd58f9fb8
# ╠═17fe2f69-2610-431f-af05-434e5ec1e6ef
# ╠═7d50fa1c-77b2-4a48-b825-1c3664684f42
# ╟─b22b3bb3-ea6b-41e5-bd74-62f5145c4416
# ╠═9113278a-604c-44fc-a51a-610c4ce37064
# ╠═aaf1e4b6-72bd-4385-a79d-ce8759f1d90a
# ╠═f0cd7ffb-4f3e-4e3a-a6cb-4f3582b9094f
# ╠═d7986e82-1825-4abc-9a0e-0f2fb9ed80fb
# ╟─76b2416d-fd48-4189-842b-157ebea1ed97
# ╟─84a429d3-653b-4572-8458-b534dd415899
# ╠═c84c8d42-152d-4e6e-bf06-099774678e50
# ╠═6760f6ec-0f99-4ef6-9752-3bcb46150955
# ╠═32fe8106-97ed-4a48-8980-e221a0ea79fb
# ╟─9d893067-18be-4982-9b42-f18de0a0ed5f
# ╠═26dde9f9-ad73-454c-b475-dfc09894f400
# ╠═53503b2d-e4c7-4570-8262-20e8ffb50156
# ╠═aa772ca5-cedd-4805-add7-0404048b6e2e
# ╟─bfc09914-c84d-4e70-8ef3-57e201c74b96
# ╠═7eb478b5-6103-44af-bd28-48e5f2fd9080
# ╠═26f71e4e-34df-44a6-89e3-e2e40db03cf7
# ╠═bc4edeec-229e-48e5-b29a-a6bafb8040c2
# ╠═579c4514-ae8e-4321-9aef-6977f2dacba2
# ╠═80f1f0c8-d4b8-4d49-a532-d73a477046c7
# ╟─5dd78dbc-5e34-4c03-bd9a-f6c1de673106
# ╠═a4701191-219f-4c3f-97f4-8dc4b35dd410
# ╠═a407201e-5260-49ea-b48f-376b1714169b
# ╠═67400cd2-9892-4502-a5de-19fdecd9598a
# ╠═027d26f8-5177-4be1-b333-a7810792c6e7
# ╠═d6c1628b-d144-4cd1-95d1-9ae7dd7e5d39
# ╟─9d3c1583-eac0-4d38-9a86-d7db17865b6f
# ╠═4d50430f-1fa0-4eee-8cf1-ef81e56f7402
# ╠═064e0ac4-7458-4a19-8e12-0395eb92a654
# ╠═704e6473-b07d-45a9-bf72-394da0ccdefe
# ╠═c5056e29-9545-4a94-bc65-6d9fd562424e
# ╟─f044966e-132e-4539-acc7-09e009b9ad4d
# ╠═6cf11a9a-a837-457d-8649-afbdf8926304
# ╟─34809859-74e2-410b-b623-6212def3cc53
# ╟─591b6955-22d5-4b30-9cc6-c2fa57adbc92
# ╠═ea52f75b-8893-41c1-b53b-0b706dc1f56a
# ╟─a060652c-9af9-4377-acd5-11b357529aca
# ╠═b908bc5c-c63d-4f00-9471-28eb51cc59fa
# ╟─7518ed7e-9f45-4cdc-b1d4-6fa6cf30b609
# ╠═75ec9e54-15ae-40f1-b249-caccc583af99
# ╠═601d2031-8b6f-481a-8e2a-e5a3af97878a
# ╠═fdf5f45c-43db-4813-adac-242b554fe9af
# ╟─11f9c8f5-e0ef-4dc0-82b0-17e20ecc88c5
# ╟─a7b566ab-16f6-4ae4-8340-3e6d3a155215
# ╟─2f2b1b7f-05a0-449a-bc29-0d2204fdd2f6
# ╠═1d746ebb-03db-4bfb-a10b-ffc0a2a3b124
# ╟─97587d11-3cab-47d8-bd6f-779195da40a5
# ╟─284be63b-61e9-45f9-8bde-0a654107d0eb
# ╠═cb332288-3b87-4dd7-8041-d1caea555879
# ╠═ec4cc343-7bbc-4824-ab1f-9d19629331ed
# ╟─be317312-cf38-4171-b2da-3d1282179088
# ╟─dbc582ea-d72c-498b-8cb4-0d19b6d87458
# ╟─36b9e359-c861-43fc-9a87-19e30afd5b24
# ╟─9f9c29a9-5213-4547-8c67-be914a39be15
# ╠═bd12b86d-01fe-454a-b1af-05b3bc7f8754
# ╟─cb7d9e99-c183-42b0-a345-69d4a89416e7
# ╟─3baed477-300c-491f-b88c-64427f8f42a6
