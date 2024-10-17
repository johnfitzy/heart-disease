### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 8f94264c-b218-40b1-b8c1-180cc4019d97
begin
using Pkg
Pkg.activate("../../") 
end

# ╔═╡ 542578f6-8823-11ef-3f6c-bd5a89644fa6
using DataFrames, CSV,  MLJ, MLJLinearModels, Plots, StatsPlots, Random,  HypothesisTests, MLJMultivariateStatsInterface

# ╔═╡ e1a48d14-037a-4331-b13a-3b229214c389
include("../../src/data_clean.jl")

# ╔═╡ 0c2dcc21-4093-4a05-9add-d79cf8ad1b6a
md"""
# Logistic Regression - DS1 heart.csv
"""

# ╔═╡ 55d0e339-e2a5-413b-a766-f9051c543db3
begin
	# Open file as CSV
	df = CSV.read("../../data/DS1/heart.csv", DataFrame) 
end

# ╔═╡ 2623efd6-f1fc-48a8-b5d0-dad6436aebe0
# Prepare X and y data
begin	

	# X features coerce to scitype Continuous as that's what the model wants
	# y targes coerce to scitype OrderdFactor as that's what the model wants
	X, y = coerce_features_and_target_to_scitypes(df)
end

# ╔═╡ 319d966e-d2cd-42e8-bcc2-fe9d379ab054
md"""
# Parameter Tuning
### Find the best parameters for the model
"""

# ╔═╡ d7bdaf56-c52a-4847-8d8f-85eb6374c7cf
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

# ╔═╡ f5b1309b-3bb7-47cb-beb3-212823d071be
begin
	
	train, test = partition(eachindex(y), 0.7, rng=1) # 70:30 split

	model = LogisticClassifier()

	# Range for the strength of the regularizer (lambda)
	r_lambda = range(model, :lambda, lower=0.0001, upper=1.0)

	# Range for penalty (regularization type: L1, L2, etc.)
	r_penalty = range(model, :penalty, values=[:l1, :l2, :en])

	# Model tunning
	tuning_model = TunedModel(
    	model=model,
    	tuning=Grid(resolution=10), # Number of distinct values in range to test
    	resampling=CV(nfolds=6),
    	ranges=[r_lambda, r_penalty],
    	measure=cross_entropy # loss function
	)

	# Create the machine and fit it
	mach = machine(tuning_model, X[train, :], y[train])
	MLJ.fit!(mach)
	
	# Report best parameter settings
	rep = report(mach)	
end

# ╔═╡ 782e5f53-f4c5-4596-b259-5165d0744a45
md"""
# Best Model
"""

# ╔═╡ d5eb0a1b-39ce-44b8-bc6d-dbabb63622f4
begin

	best_model = rep.best_model
	
	# Create a machine with the best model
	best_model_mach = machine(best_model, X[train, :], y[train])

	# Fit it
	MLJ.fit!(best_model_mach)

	# Predict probabilities for the positive class
	y_pred_prob = predict(best_model_mach, X[test, :])

	# Convert probabilities to hard class predictions (0 or 1)
	y_pred_best = predict_mode(best_model_mach, X[test, :])
end

# ╔═╡ 83226fd9-8274-436c-b169-9b196d3a3d3d
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using JLSO
	MLJ.save("../../models/Logistic_Classifier_DS1_Model.jls", best_model_mach)
end
  ╠═╡ =#

# ╔═╡ eb4a9547-b2dd-4391-8309-672ab33dedcd
md"""
# Performance Evaluation of Best Model
### Confusion Matrix
"""

# ╔═╡ 62a45d09-6703-4bec-9aff-6f22fbd51ac3
ConfusionMatrix()(y_pred_best, y[test])

# ╔═╡ a33ede6f-0d68-424b-aae0-64554f14d987
md"""
### ROC Curve
"""

# ╔═╡ 83e28242-8e71-4580-83df-6441513f0421
begin
	# Plot ROC 
	fprs, tprs, thresholds = roc_curve(y_pred_prob, y[test])

	plot(fprs, tprs, label="Logistic Classifier", xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve for Logistic Classifier")

end

# ╔═╡ 5e07681d-c0fd-45ad-ae5b-238c60247f9d
begin
	@info "Model accuracy: $(MLJ.accuracy(y_pred_best, y[test]))"
	@info "AUC: $(auc(y_pred_prob, y[test]))"
end

# ╔═╡ a259f983-8888-4363-92d7-7accac42e0d7
md"""
# Comments:
## Comment on Confusion Matrix:
- True Positives (TP) for class 1: 29 instances of class 1 were correctly predicted as class 1.
- False Positives (FP) for class 1: 6 instances were incorrectly predicted as class 1 but actually belong to class 0.
- True Negatives (TN) for class 0: 36 instances of class 0 were correctly predicted as class 0.
- False Negatives (FN) for class 0: 10 instances were incorrectly predicted as class 0 but actually belong to class 1.

## Model Accuracy:
- Model accuracy: 0.802
  - This means the model correctly predicted approximately 80.2% of the total instances.

## ROC Curve and AUC:
- The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
- A higher ROC curve indicates better performance. The closer the curve is to the top left corner, the better the model is at distinguishing between the classes.
- The AUC (Area Under the Curve) summarizes the ROC curve performance; a value of 1.0 represents perfect classification, while a value of 0.5 represents random guessing.
- The AUC is 0.911, which indicates strong model performance.
"""


# ╔═╡ f6bea807-5dbe-47d5-9da4-2db6a12a2494
md"""
# Visualise
### Plot against PCA
"""

# ╔═╡ e2d6811b-7184-4d84-8b9c-1b0f08d0c6bb
PCA_model = @load PCA pkg=MultivariateStats

# ╔═╡ f6070d26-5326-4eee-9a71-518f7eb4e22c
begin

	pca_model = PCA_model(maxoutdim=2)  # Reducing to 2 principal components
	pca_mach = machine(pca_model, X[test, :])
	MLJ.fit!(pca_mach)


	X_pca = MLJ.transform(pca_mach, X[test, :])

	scatter(X_pca[:, 1], 
		X_pca[:, 2], 
		group=y[test], 
		legend=:topright, 
		xlabel="PC1", 
		ylabel="PC2",
   	 	title="PCA plot of features after Basic Logistic Regression")
end

# ╔═╡ 3079d67b-a33b-4110-9e8c-6bedafb62853
md"""
## Serialize model for later use
### Purposely disabled.
"""

# ╔═╡ a314e13e-ae44-4575-bed2-60839126509a
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	# Restore the model -  testing
	restored_mach = machine("../../models/Logistic_Classifier_DS1_Model.jls")
end
  ╠═╡ =#

# ╔═╡ 85006dcc-f2fe-4b87-8d64-d0a51b00dfec
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	y_pred_bestold = predict_mode(best_model_mach, X[test, :])
	y_pred_bestnew = predict_mode(restored_mach, X[test, :])
	y_pred_bestold == y_pred_bestnew
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═0c2dcc21-4093-4a05-9add-d79cf8ad1b6a
# ╠═8f94264c-b218-40b1-b8c1-180cc4019d97
# ╠═542578f6-8823-11ef-3f6c-bd5a89644fa6
# ╠═e1a48d14-037a-4331-b13a-3b229214c389
# ╠═55d0e339-e2a5-413b-a766-f9051c543db3
# ╠═2623efd6-f1fc-48a8-b5d0-dad6436aebe0
# ╠═319d966e-d2cd-42e8-bcc2-fe9d379ab054
# ╠═d7bdaf56-c52a-4847-8d8f-85eb6374c7cf
# ╠═f5b1309b-3bb7-47cb-beb3-212823d071be
# ╠═782e5f53-f4c5-4596-b259-5165d0744a45
# ╠═d5eb0a1b-39ce-44b8-bc6d-dbabb63622f4
# ╟─eb4a9547-b2dd-4391-8309-672ab33dedcd
# ╠═62a45d09-6703-4bec-9aff-6f22fbd51ac3
# ╠═a33ede6f-0d68-424b-aae0-64554f14d987
# ╠═83e28242-8e71-4580-83df-6441513f0421
# ╠═5e07681d-c0fd-45ad-ae5b-238c60247f9d
# ╟─a259f983-8888-4363-92d7-7accac42e0d7
# ╠═f6bea807-5dbe-47d5-9da4-2db6a12a2494
# ╠═e2d6811b-7184-4d84-8b9c-1b0f08d0c6bb
# ╠═f6070d26-5326-4eee-9a71-518f7eb4e22c
# ╠═3079d67b-a33b-4110-9e8c-6bedafb62853
# ╠═83226fd9-8274-436c-b169-9b196d3a3d3d
# ╠═a314e13e-ae44-4575-bed2-60839126509a
# ╠═85006dcc-f2fe-4b87-8d64-d0a51b00dfec
