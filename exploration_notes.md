# Exploration Notes

## Data Attribute Description:

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

## Data Sneak Peak:

| age  | sex  | chest_pain | rest_bp | serum_chol | fasting_blood_sugar | electrocardiographic | max_heart_rate | angina | oldpeak | slope | major_vessels | thal | heart_disease |
|------|------|------------|---------|------------|---------------------|----------------------|----------------|--------|---------|-------|---------------|------|---------------|
| 70.0 | 1.0  | 4.0        | 130.0   | 322.0      | 0.0                 | 2.0                  | 109.0          | 0.0    | 2.4     | 2.0   | 3.0           | 3.0  | 2             |
| 67.0 | 0.0  | 3.0        | 115.0   | 564.0      | 0.0                 | 2.0                  | 160.0          | 0.0    | 1.6     | 2.0   | 0.0           | 7.0  | 1             |
| 57.0 | 1.0  | 2.0        | 124.0   | 261.0      | 0.0                 | 0.0                  | 141.0          | 0.0    | 0.3     | 1.0   | 0.0           | 7.0  | 2             |

From the above we can see some initial properties of the data
2) Features are Categorical, Continuous and Binary types
1) Many of the features are in different scales
## Feature Analysis

### Age (Continuous)
Summary Stats:
```
Mean:           54.433333
Std. Deviation: 9.109067
Minimum:        29.000000
1st Quartile:   48.000000
Median:         55.000000
3rd Quartile:   61.000000
Maximum:        77.000000
```
- The data closely follows a normal distribution, as shown by the QQ Plot, although there are some outliers at the tails, indicating deviations at the extremes of the age spectrum.
- The violin plot suggests a slight negative skew, with a tendency toward older individuals, implying that ages above the mean are slightly more prevalent.
- A noticeable bump around the 40-45 year range suggests a local concentration of data points, which may reflect a particular subgroup or trend within the data.
- The standard deviation of 9.109067 is about 16.7% of the mean (54.433333), indicating moderate variability. This suggests that while there is some spread in the ages, most data points remain relatively close to the mean, reflecting a reasonably consistent age distribution with moderate dispersion.
- The proximity of the median (55) to the mean further supports the observation of a near-symmetrical distribution, despite the slight skew and presence of outliers.

### Sex (Binary)
Summary Stats:
- 67.8 % are male
- 32.2 % are female

### Chest Pain Type (Categorical)
Summary Stats:
- Typical Angina 7.4%
- Atypical Angina 15.6%
- Non-Anginal Pain 29.3%
- Asymptomatic 47.8%

### Resting Blood Pressure (Continuous)

Summary Stats:
```
Mean:           131.344444
Std. Deviation: 17.861608
Minimum:        94.000000
1st Quartile:   120.000000
Median:         130.000000
3rd Quartile:   140.000000
Maximum:        200.000000
```
- The probability distribution is somewhat close to a normal distribution but exhibits a slight positive skew, with lower blood pressure readings being more frequent and a longer right tail indicating higher values.
- The data contains several outliers, with some very high readings indicating elevated blood pressure and a few low outliers. These extremes suggest the presence of individuals with unusual resting blood pressure levels, both high and low.
- The standard deviation of 17.861608 is about 13.6% of the mean (131.344444), indicating moderate variability relative to the mean. This suggests that the data points are relatively close to the mean, reflecting a reasonably tight distribution compared to the overall scale of blood pressure values.
- The median (130) is close to the mean, which indicates a reasonably symmetric central tendency, despite the presence of skew and outliers.
- The interquartile range (120 to 140) captures the middle 50% of the data, showing a typical range of resting blood pressure values that are relatively close to the average, supporting the observation of moderate variability.

### Serum cholesterol (Continuous)
Summary Stats:
```
Mean:           249.659259
Std. Deviation: 51.686237
Minimum:        126.000000
1st Quartile:   213.000000
Median:         245.000000
3rd Quartile:   280.000000
Maximum:        564.000000
```
- The probability distribution is somewhat close to a normal distribution; however, the presence of high cholesterol outliers creates a significant positive skew, resulting in a long right tail. This suggests that while most cholesterol levels are centered around the mean, there are individuals with unusually high levels.
- The standard deviation of 51.686237 is approximately 20.7% of the mean (249.659259), indicating moderate variability in relation to the average cholesterol level. This suggests that the data points are fairly spread around the mean, reflecting moderate dispersion compared to the typical value.
- The median (245) is slightly lower than the mean, reinforcing the presence of skew toward higher values due to the outliers.
- The interquartile range (213 to 280) shows where the central 50% of the data lies, which is relatively close to the mean but skewed upward, highlighting that while typical values are consistent, extreme values contribute significantly to the overall spread.
- Overall, the data reflects a distribution where most individuals have cholesterol levels close to the mean, but the influence of high outliers pulls the average up, leading to the observed skew and extended tail.


### Fasting Blood Sugar (Binary)
- 14.8% of people had elevated fasting blood sugar levels
- 85.2% of people had normal fasting blood sugar levels

### Resting ECG Results (Categorical) 
- Normal 7.4%
- ST-T Wave Abnormality 15.6%
- Left Ventricular Hypertrophy 29.3%


### Maximum Heart Rate Achieved (Continuous)
Summary Stats:
```
Mean:           149.677778
Std. Deviation: 23.165717
Minimum:        71.000000
1st Quartile:   133.000000
Median:         153.500000
3rd Quartile:   166.000000
Maximum:        202.000000
```
- The data exhibits a negative skew with a long left tail, indicating that there are outliers with unusually low maximum heart rates. This skew suggests that while most individuals achieve higher maximum heart rates, some have significantly lower values that pull the distribution leftward.
- The standard deviation is 23.165717, which is approximately 15.5% of the mean (149.677778). This indicates moderate variability relative to the average, suggesting that the heart rate values are spread moderately around the mean, reflecting a fairly consistent range for most individuals.
- The median (153.5) is slightly higher than the mean, further supporting the negative skew, as more data points lie above the average, with fewer, but more extreme, lower values.
- The interquartile range (133 to 166) captures the central 50% of the data, indicating that most maximum heart rates are relatively close to the mean, despite the presence of lower outliers.
- Overall, the data suggests that while most maximum heart rates are clustered around typical values, the left skew highlights the influence of individuals with notably lower heart rates, contributing to the overall spread and shape of the distribution.

### Exercise Induced Angina (Binary) 
- 33.0% have exercise induced angina
- 77.0% do not have exercise induced angina

### ST Depression Induced by Exercise Relative to Rest - oldspeak (Continuous)
Summary Stats:
```
Mean:           1.050000
Std. Deviation: 1.145210
Minimum:        0.000000
1st Quartile:   0.000000
Median:         0.800000
3rd Quartile:   1.600000
```

- The data is heavily positively skewed, with a long right tail, indicating that most values are concentrated at lower levels of ST depression, while a few cases exhibit significantly higher values, extending the distribution to the right.
- The standard deviation is 1.145210, which is approximately 109% of the mean (1.050000), indicating high variability relative to the average value. This large spread suggests that while many individuals have low ST depression levels, there is considerable dispersion with some experiencing much higher values.
- The median (0.8) is below the mean, highlighting the positive skew and indicating that over half of the observations fall below the average value, with outliers on the higher end pulling the mean upward.
- The interquartile range (0.0 to 1.6) suggests that the central 50% of the data is relatively concentrated at the lower end, emphasizing that most individuals experience minimal to moderate levels of ST depression.
- Overall, the distribution reflects a pattern where lower values are most common, but the presence of high outliers creates a long tail, driving the overall skew and variability observed in the data.

### The Slope of the Peak Exercise ST Segment - slope (Category) 

- Upsloping 7.4%
- Flat 15.6%
- Downsloping 29.3%

### Number of Major Vessels (0-3) Colored by Fluoroscopy - major-vessels (Continuous)

Summary Stats:
```
Mean:           0.670370
Std. Deviation: 0.943896
Minimum:        0.000000
1st Quartile:   0.000000
Median:         0.000000
3rd Quartile:   1.000000
Maximum:        3.000000
```

- The data is heavily positively skewed with a long right tail, indicating that the majority of observations have zero major vessels colored by fluoroscopy, while a smaller number of cases have high-er values.
- The standard deviation is 0.943896, which is approximately 141% of the mean (0.670370). This high relative variability suggests a wide spread compared to the average value, primarily driven by the presence of higher values and the skewed nature of the data.
- The median (0.000000) and the first quartile (0.000000) indicate that at least half of the data points have no major vessels colored, underscoring the concentration of data at the lower end.
- The third quartile (1.000000) and maximum (3.000000) highlight the extent of variability, with a smaller proportion of individuals having more vessels involved, which significantly contributes to the long right tail.
- Overall, the distribution reflects a dataset where the majority of cases have no vessels colored, but a minority with higher counts creates a substantial skew, pulling the mean up and increasing overall variability.

### Thal (Categorical) 3 = normal; 6 = fixed defect; 7 = reversible defect 
- Normal 7.4%
- Fixed Defect 15.6%
- Reversible Defect 29.3%
