### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 8f280225-9419-4d42-9b81-138f876bb21e
begin
using Pkg
Pkg.activate("../../") 
end

# ╔═╡ c51ca81e-8865-11ef-013e-4b6b7c978355
using DataFrames, CSV,  MLJ, MLJLinearModels, Plots, StatsPlots, Random,  HypothesisTests, MLJMultivariateStatsInterface

# ╔═╡ 6479e71e-667d-479e-a165-32791c693a62
include("../../src/data_clean.jl")

# ╔═╡ 4d488560-f97e-47cb-b7ec-b06129cafed4
md"""
# Logistic Regression - DS2 imputed.processed.hungarian.csv
"""

# ╔═╡ ac0eee90-3679-472a-97be-3d489fc5fa7b
df = CSV.read("../../data/DS2_imputed/imputed.processed.hungarian.csv", 
	delim=',', 
	DataFrame) 

# ╔═╡ 184762f2-df9a-495e-870b-05af14f9afbf
# X features coerce to scitype Continuous as that's what the model wants
# y targes coerce to scitype OrderdFactor as that's what the model wants
X, y = coerce_features_and_target_to_scitypes(df)

# ╔═╡ 86f22f7a-486d-4ff0-ae1e-0a217e5c2766
md"""
# Parameter Tuning
### Find the best parameters for the model
"""

# ╔═╡ b9d8eb74-cf88-48f4-b501-79327217a9b6
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

# ╔═╡ f610d305-f297-4ac8-8656-176f547e5304
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
    	measure=cross_entropy
	)

	# Create the machine and fit it
	mach = machine(tuning_model, X[train, :], y[train])
	MLJ.fit!(mach)

	# Report best parameter settings
	rep = report(mach)	
end

# ╔═╡ 72164307-1a47-4591-a42d-7d1a11003f93
md"""
# Best Model
"""

# ╔═╡ 51680451-2619-4e9c-9d25-6854bc3a1939
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

# ╔═╡ 9a8d3e58-be76-4cce-aa11-dffa21910bc9
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using JLSO
	MLJ.save("../../models/Logistic_Classifier_DS2_Hungarian_Model.jls", best_model_mach)
end
  ╠═╡ =#

# ╔═╡ 3e6f38c4-2a11-4417-b530-aa23ea3bd280
md"""
# Performance Evaluation of Best Model
### Confusion Matrix
"""

# ╔═╡ 283b229f-7f05-4899-9e79-778d98646d68
ConfusionMatrix()(y_pred_best, y[test])

# ╔═╡ ff3e558c-a1bc-4a27-be3c-7615e5110e13
md"""
### ROC Curve
"""

# ╔═╡ f054dfc7-a99b-40ad-88ef-ab1501f03ecc
begin
	# Plot ROC 
	fprs, tprs, thresholds = roc_curve(y_pred_prob, y[test])

	plot(fprs, tprs, label="Logistic Classifier", xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve for Logistic Classifier")

end

# ╔═╡ e72c7379-56b3-4e20-8ebf-697952ff8081
begin
	@info "Model accuracy: $(MLJ.accuracy(y_pred_best, y[test]))"
	@info "AUC: $(auc(y_pred_prob, y[test]))"
end

# ╔═╡ 1f22734e-8790-48a7-8156-606b4a703499
md"""
# Comments:
## Comment on ConfusionMatrix:
- True Positives (TP) for class 0: 52 instances of class 0 were correctly predicted as class 0.
- False Positives (FP) for class 0: 6 instances were incorrectly predicted as class 0 but actually belong to class 1.
- True Negatives (TN) for class 1: 20 instances of class 1 were correctly predicted as class 1.
- False Negatives (FN) for class 1: 10 instances were incorrectly predicted as class 1 but actually belong to class 0.

## Model Accuracy:
- Model accuracy: 0.818
  - This means the model correctly predicted 81.8% of the total instances.

## ROC Curve and AUC:
- The ROC (Receiver Operating Characteristic) curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
- A higher ROC curve indicates better performance. The closer the curve is to the top left corner, the better the model is at distinguishing between the classes.
- The AUC (Area Under the Curve) summarizes the ROC curve performance; a value of 1.0 represents perfect classification, while a value of 0.5 represents random guessing.
- The AUC for this model is 0.845, which indicates reasonable performance in distinguishing between the classes.


"""

# ╔═╡ 7a5766b9-a1e0-42b0-b371-8a9aba545169
md"""
# Visualise
### Plot against PCA
"""

# ╔═╡ 3e44a8cc-1132-4ea7-a361-931395fe3791
PCA_model = @load PCA pkg=MultivariateStats

# ╔═╡ 75c70dbe-a4da-458b-9bee-d007997858e8
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

# ╔═╡ 587cef9d-aa10-4c2a-98f9-1efa3c0017c9
md"""
## Serialize model for later use
### Purposely disabled.
"""

# ╔═╡ 950ebb3c-3d25-482b-b35c-0e9bdedf58cc
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	# Restore the model -  testing
	restored_mach = machine("../../models/Logistic_Classifier_DS2_Hungarian_Model.jls")
end
  ╠═╡ =#

# ╔═╡ b55fccfb-8737-4dba-bdb5-43a0eb232482
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
# ╟─4d488560-f97e-47cb-b7ec-b06129cafed4
# ╠═c51ca81e-8865-11ef-013e-4b6b7c978355
# ╠═8f280225-9419-4d42-9b81-138f876bb21e
# ╠═6479e71e-667d-479e-a165-32791c693a62
# ╠═ac0eee90-3679-472a-97be-3d489fc5fa7b
# ╠═184762f2-df9a-495e-870b-05af14f9afbf
# ╠═86f22f7a-486d-4ff0-ae1e-0a217e5c2766
# ╠═b9d8eb74-cf88-48f4-b501-79327217a9b6
# ╠═f610d305-f297-4ac8-8656-176f547e5304
# ╠═72164307-1a47-4591-a42d-7d1a11003f93
# ╠═51680451-2619-4e9c-9d25-6854bc3a1939
# ╠═3e6f38c4-2a11-4417-b530-aa23ea3bd280
# ╠═283b229f-7f05-4899-9e79-778d98646d68
# ╠═ff3e558c-a1bc-4a27-be3c-7615e5110e13
# ╠═f054dfc7-a99b-40ad-88ef-ab1501f03ecc
# ╟─e72c7379-56b3-4e20-8ebf-697952ff8081
# ╠═1f22734e-8790-48a7-8156-606b4a703499
# ╠═7a5766b9-a1e0-42b0-b371-8a9aba545169
# ╠═3e44a8cc-1132-4ea7-a361-931395fe3791
# ╠═75c70dbe-a4da-458b-9bee-d007997858e8
# ╠═587cef9d-aa10-4c2a-98f9-1efa3c0017c9
# ╠═9a8d3e58-be76-4cce-aa11-dffa21910bc9
# ╠═950ebb3c-3d25-482b-b35c-0e9bdedf58cc
# ╠═b55fccfb-8737-4dba-bdb5-43a0eb232482
