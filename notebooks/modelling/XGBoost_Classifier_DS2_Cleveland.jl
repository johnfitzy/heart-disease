### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 9850ab27-22e5-47c2-83d5-a29d25ae8c7e
begin
using Pkg
Pkg.activate("../../") 
end

# ╔═╡ c10fb950-84cd-11ef-2b27-e7f80c61964d
using DataFrames, Plots, MLJ, MLJModels, HypothesisTests, CSV, Printf, Plots, MLJLinearModels, StatsPlots, Statistics, CategoricalArrays, MLJXGBoostInterface, Random, XGBoost

# ╔═╡ 5034f5fc-0328-40b6-aa24-176ad10d6370
include("../../src/data_clean.jl")

# ╔═╡ f75fa5da-cca1-400f-8d2b-0c88c5f20c6e
md"""

# Gradient Boosting: XGBoost Classifier model on Imputed DS2 Cleveland

- This book runs as follows:

1. Clean and flatten dataset as per spec.
2. Initial build of XGBoost Classifier machine
3. Train on partitioned DS2 data
4. Performance metrics: Log loss, Accuracy, Confusion Metrics, violin plots etc.
5. Autotune to find best performance hyperparameters
6. Applied new hyperparameters to rebuild a tuned model
7. Performance metrics on tuned model
8. T-test/Mann-Whitney Tests between untuned and tuned model
9. Summary of XGBoost
"""

# ╔═╡ 6f3ecc26-8800-41ee-ad24-d4e6bb9a2ed9
df= CSV.read("../../data/DS2_imputed/imputed.processed.cleveland.csv", DataFrame)

# ╔═╡ 89e80d11-27d9-4461-ba54-9e8468eedeb7
begin
X = DataFrames.select(df, Not(:heart_disease))
y = categorical(df.heart_disease)
X, y = coerce_features_and_target_to_scitypes(df)	
end

# ╔═╡ fa61bef0-b252-45a2-96d4-1a74c49c1002
begin
train, test = partition(1:length(y), 0.7, shuffle=true)
X_train = X[train, :]
y_train = y[train]
X_test = X[test, :]
y_test = y[test]
end

# ╔═╡ 564c890b-f8a0-43b9-8c15-11bd2522902a
begin
dtrain = DMatrix(X_train, label=y_train)
dtest = DMatrix(X_test, label=y_test)
end

# ╔═╡ 5c9dd70e-5124-45c9-826b-7b97d3b72ca7
md"""
# XGBoost Classifier Ensemble
"""

# ╔═╡ b70ab936-f767-49ce-95e4-32e8da8b2686
XGBoostClassifier = @load XGBoostClassifier pkg=XGBoost

# ╔═╡ ca1f1fd8-345b-4157-9d6a-96921cb0d06f
xgb =XGBoostClassifier(objective="multi:softmax")

# ╔═╡ 4bee6d6a-7195-4ff9-8a7a-93e8f15d3ea6
begin
mach = machine(xgb, X, y, scitype_check_level=0)
MLJ.fit!(mach, rows=train, verbosity=0)
end

# ╔═╡ 3623daea-abdb-42e1-a05d-81b61f19b111
xgb

# ╔═╡ 41dfa78e-7538-4f78-bfb6-10772a1345d4
md"""
## Performance Evaluation of Untuned Model

- Accuracy, Confusion Matrix, ROC/AUC, Violin Plot, Learning Curve
"""

# ╔═╡ 43d7fa91-1384-4831-a91f-58b03dc251ba
ypred= vcat(predict_mode(mach, dtest)...)

# ╔═╡ 00d8a634-b538-4c94-bf3d-38e5889d51b8
ypredprob= MLJ.predict(mach,dtest)

# ╔═╡ 5561756d-9b2b-4253-9004-cbcc191c4044
md"""
### Accuracy""" 

# ╔═╡ 4991da63-0a38-4f80-832a-dbb3a889a8a1
accu_U=round(accuracy(ypred, y_test)*100, digits=2)

# ╔═╡ 1b1f1805-7490-406c-84da-94d72da992ef
begin
y_pred_1= categorical(ypred, ordered=true)
y_test_1= categorical(y_test, ordered=true)
end

# ╔═╡ e799a120-be6e-4226-9c50-a84f7662cf5a
md"""
### Confusion Matrix of Untuned Model"""

# ╔═╡ beedb402-eca8-43b4-b021-d3a71683f05b
cm_U= ConfusionMatrix()(y_pred_1, y_test_1)

# ╔═╡ 08bd5327-4420-4cf0-823c-0d798e72a131
md"""
### ROC Curve"""

# ╔═╡ 207227eb-05db-49cd-aec6-31acd0b7a2fd
begin
	# Plot ROC 
	fprs, tprs, thresholds = roc_curve(ypredprob, y_test)

	plot(fprs, tprs, label="XGBoost Classifier", legend=:bottomright,xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve for XGBoost Classifier")

end

# ╔═╡ 26652084-e92a-474c-8ebc-f9a9ec0721e9
auc_u= round(auc(ypredprob, y_test), digits=3)

# ╔═╡ e965bc04-945b-4e05-9465-91670406e5bd
trtime=@timed begin
	Random.seed!(1)
	perf = MLJ.evaluate!(mach, measure=log_loss, resampling=CV(nfolds=10, shuffle=true), check_measure=false, verbosity=0)

end

# ╔═╡ c3d13c29-569c-4938-9c6a-5a538fe57d3a
XGperf= vcat(vcat(perf.per_fold...)...)

# ╔═╡ 4be1ed23-02e8-4adc-8699-383db459a295
violin(["XGBoost Regression log loss"], XGperf, label=nothing)

# ╔═╡ a0a34df2-0653-4a3b-8f69-aa15a959b4ad
begin
meanXG=round(mean(XGperf), sigdigits=4)
stdXG=round(std(XGperf), sigdigits=4)
println("mean XG log loss: $(meanXG)")
println("standard deviation of XG log loss: $(stdXG)")
end

# ╔═╡ 36f403b0-f576-4868-9b5d-8855bf3e4c90
begin
	r_n = range(xgb, :num_round, lower=1, upper=100)
	curve = learning_curve(mach, resampling=CV(nfolds=10),
	                         range=r_n, measure=log_loss);
end

# ╔═╡ 9de2c271-4e4c-4287-8c7f-e9643799db33
begin
plotly()
Plots.plot(curve.parameter_values, curve.measurements, ylabel="log loss", xlabel="# num_round", label="", title="Num. of Boosting rounds vs log loss")
end

# ╔═╡ f9479004-8352-4a99-8e55-37e63d56b898
md"""
# Summary and Comments on Untuned XGBoost Model

## Summary:
- The untuned model Accuracy is: $accu_U %
- AUC: $auc_u
- mean XG log loss: $(meanXG)
- standard deviation of XG log loss: $(stdXG)

## Comments:

### ROC/AUC:
- The AUC (Area Under the Curve) summarizes the ROC curve performance; a value of 1.0 represents perfect classification, while a value of 0.5 represents random guessing.
- The AUC is $auc_u, which is not good (0.5=random guessing)


"""

# ╔═╡ 738b39e7-d429-458c-b843-2de6358046f2
cm_U

# ╔═╡ 41883cf3-e8f9-4192-9d65-0b3410d96e0b
md"""
# Autotuning the XGBoost Classifier
"""

# ╔═╡ 6bc31402-4be2-4810-aacd-c3a6fd53d69f
#autotuning 
begin
	Random.seed!(1)
	r1 = range(xgb, :eta, lower=0.1, upper=0.3)
	r2 = range(xgb, :num_round, lower=2, upper=20)
	tm = TunedModel(model=xgb, tuning=Grid(resolution=10),
	                resampling=CV(nfolds=10, shuffle=true), ranges=[r1, r2],
	                measure=log_loss)
	m = machine(tm, X, y, scitype_check_level=0)
	MLJ.fit!(m, rows=train, verbosity=0);
end

# ╔═╡ 15934c79-9379-41e3-94c2-76275779e560
begin
	rep = report(m)
	rep.best_history_entry.measurement
end

# ╔═╡ 20057c27-6120-4a56-a316-b29dfcb776a4
rep

# ╔═╡ a22e901e-d71f-47f9-89b3-69c786c977e1
begin
bnumround= rep.best_history_entry.model.num_round
beta= round((rep.best_history_entry.model.eta), sigdigits=3)
Log_loss= round((rep.best_history_entry.measurement[1]), sigdigits=3)
end

# ╔═╡ 5640eb69-ac29-4de0-b3e8-5ba504bda87f
md"""
## Best settings: 

- num_round = $bnumround
- eta = $beta
- Log loss = $Log_loss (lower is better)

	"""

# ╔═╡ 5b2b4a34-e4c4-41d4-8720-2afa9480765f

begin
	res = rep.plotting
	vals_sf = res.parameter_values[:, 1]
	vals_df = res.parameter_values[:, 2];
end

# ╔═╡ eda5fb91-5669-4a31-bfce-0335842f6df6
res

# ╔═╡ 77e159d5-1f79-4e04-b067-b53eb2583ed7
begin
	
	p=Plots.contour(vals_sf, vals_df, res.measurements, cmap=:hot, fill=true)
	xlabel!("# of num_round", fontsize=14)
	ylabel!("eta", fontsize=14)
	title!("Contour heatmap for best setting")
	p
end

# ╔═╡ 1d6bb089-6f64-4777-9922-be3ff8a01af8
md""" 
## Tuned XGBoost Model with tuned hyperparameters
- including performance metrics: Accuracy, Confusion Matrix, ROC/AUC, Violin Plot
- T-test and Mann-Whitney test

"""

# ╔═╡ 6b9b0b30-d17a-4cd2-9422-65345f4aecc6
begin
t_xgb = XGBoostClassifier(eta=beta, num_round=bnumround, objective="multi:softmax")
t_mach = machine(t_xgb, X, y, scitype_check_level=0)
MLJ.fit!(t_mach, rows=train, verbosity=0)
end

# ╔═╡ 3ae9822c-58a7-4e6f-a39d-6d9e88e18d2f
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	using JLSO
	MLJ.save("../../models/XGBoost_Classifier_DS2_Cleveland_Model.jls", t_mach)
end
  ╠═╡ =#

# ╔═╡ 23a37e56-a44b-4fec-b90b-5db40aa582aa
tutime=@timed begin
	Random.seed!(1)
	t_perf = MLJ.evaluate!(t_mach, measure=log_loss, resampling=CV(nfolds=10, shuffle=true), check_measure=false, verbosity=0)

end

# ╔═╡ 3431044c-8166-44a7-a6fc-cdd35a72e79b
XGperfT= vcat(vcat(perf.per_fold...)...)

# ╔═╡ b996450c-0407-48ae-982e-782c1fc796da
begin
Utime, Ubyte= round((trtime.time), sigdigits=4), round((trtime.bytes/1000000), sigdigits=4)
Ttime, Tbyte= round((tutime.time), sigdigits=4), round((tutime.bytes/1000000), sigdigits=4)
end

# ╔═╡ 1635a73a-c522-4796-a8d4-322668f1ab19
begin
loglossT= round((tutime.value.measurement[1]), digits=4)
end

# ╔═╡ ca1e599d-b173-478f-9fb7-6ce080f0ee19
begin
all_perfT=vcat(vcat(tutime.value.per_fold...)...)
stdT= round(std(all_perfT), digits=4)
meanT= round(mean(all_perfT), sigdigits=4)
end

# ╔═╡ 586e952a-e3f0-46b9-b239-8d1d7a785835
ypredT= predict_mode(t_mach, dtest)

# ╔═╡ 01eb6f8d-2787-47ff-b6c8-ccc4e7ebd3e4
ypredTprob= MLJ.predict(t_mach, dtest)

# ╔═╡ 1c99b55f-d6b4-45af-bdd4-468423f33c8d
y_test

# ╔═╡ a19d0448-baee-4a28-8f9f-3c20028e62d0
accu_T=round(accuracy(ypredT, y_test)*100, digits=2)

# ╔═╡ 7baae015-6314-4ee2-b716-cc24273a7c39
begin
y_pred_o= categorical(ypredT, ordered=true)
y_test_o= categorical(y_test, ordered=true)
end

# ╔═╡ 9e9f39b8-2202-46ff-bab7-666f803caf7f
cm_T= ConfusionMatrix()(y_pred_o, y_test_o)

# ╔═╡ 1cd26f66-cbf9-4006-8ae6-4851adb8c08d
# ╠═╡ disabled = true
#=╠═╡
begin
	# Plot ROC 
	fprsT, tprsT, thresholdsT = roc_curve(ypredTprob, y_test)

	plot(fprsT, tprsT, label="XGBoost Classifier", xlabel="False Positive Rate", ylabel="True Positive Rate", legend=:bottomright,title="ROC Curve for Tuned XGBoost Classifier")

end
  ╠═╡ =#

# ╔═╡ 7a48a728-9bad-4838-93a4-7c7ecedf4521
auc_t= round(auc(ypredTprob, y_test), digits=3)

# ╔═╡ f47372b8-d515-4f65-93e6-7a8eaa5d5902
md"""
### Comparing untune and tuned model
- violin plots side by side
- T-test 
- Mann-Whitney Test"""

# ╔═╡ c5521565-8e7a-4204-896c-cefb9685eb7d
begin
    violin(["Untuned Vs Tuned XGBoost Classifier"], XGperf, side=:left, label="Untuned")
    violin!(["Untuned Vs Tuned XGBoost Classifier"], all_perfT, side=:right, label="Tuned",title="Log loss Untuned Vs Tuned side-by-side")
end

# ╔═╡ 7853c94f-60ff-4e09-98ad-8e6468c73cf0
UnequalVarianceTTest(XGperf, all_perfT)

# ╔═╡ cae4dfc3-0b1c-4cd2-94d9-02dc1fe73553
MannWhitneyUTest(all_perfT, XGperf)

# ╔═╡ 0f241ad4-7b04-4999-bd01-d67898d474d4
md"""
## Summary:

- This book contains a tuned model of XGBoost Classifier, trained on partitioned (7:3)DS1, tuned and tested. The machine is autotuned in hyperparameters of number of rounds of boosting= $bnumround and learning rate= $beta.
- Performance metrics of the tuned model are as follows:
- mean log_loss: $meanT   (Untuned model: $meanXG) *(Lower is better)*
- standard dev: $stdT (Untuned model: $stdXG)
- Accuracy: $accu_T %  (Untuned model: $accu_U %)
- AUC: $auc_t (Untuned model: $auc_u)


- The T-test and Mann-Whitney test showed that they are not only statistically different from one but also that the tuned model performed better.

Additionally, the tuned model due to decrease in boosting rounds, uses less time and computational resources.

- Untuned model: $Utime seconds. $Ubyte Mb
- Tuned model: $Ttime seconds. $Tbyte Mb


"""



# ╔═╡ eec3e5b2-ec1c-45c2-b84f-6fd56fe6f73c
md"""
### Comments on Confusion Matrix:
As you can see below, there is improvements in the testing dataset:
"""

# ╔═╡ f990ad86-7ad9-4f19-beb3-7fed28765166
display(cm_U) #Untuned Model

# ╔═╡ b004f4d1-c28c-4df5-941f-e66d190fb37e
display(cm_T) #Tuned Model

# ╔═╡ Cell order:
# ╟─f75fa5da-cca1-400f-8d2b-0c88c5f20c6e
# ╠═9850ab27-22e5-47c2-83d5-a29d25ae8c7e
# ╠═c10fb950-84cd-11ef-2b27-e7f80c61964d
# ╠═5034f5fc-0328-40b6-aa24-176ad10d6370
# ╠═6f3ecc26-8800-41ee-ad24-d4e6bb9a2ed9
# ╠═89e80d11-27d9-4461-ba54-9e8468eedeb7
# ╠═fa61bef0-b252-45a2-96d4-1a74c49c1002
# ╠═564c890b-f8a0-43b9-8c15-11bd2522902a
# ╟─5c9dd70e-5124-45c9-826b-7b97d3b72ca7
# ╠═b70ab936-f767-49ce-95e4-32e8da8b2686
# ╠═ca1f1fd8-345b-4157-9d6a-96921cb0d06f
# ╠═4bee6d6a-7195-4ff9-8a7a-93e8f15d3ea6
# ╠═3623daea-abdb-42e1-a05d-81b61f19b111
# ╟─41dfa78e-7538-4f78-bfb6-10772a1345d4
# ╠═43d7fa91-1384-4831-a91f-58b03dc251ba
# ╠═00d8a634-b538-4c94-bf3d-38e5889d51b8
# ╟─5561756d-9b2b-4253-9004-cbcc191c4044
# ╠═4991da63-0a38-4f80-832a-dbb3a889a8a1
# ╠═1b1f1805-7490-406c-84da-94d72da992ef
# ╟─e799a120-be6e-4226-9c50-a84f7662cf5a
# ╠═beedb402-eca8-43b4-b021-d3a71683f05b
# ╟─08bd5327-4420-4cf0-823c-0d798e72a131
# ╠═207227eb-05db-49cd-aec6-31acd0b7a2fd
# ╠═26652084-e92a-474c-8ebc-f9a9ec0721e9
# ╠═e965bc04-945b-4e05-9465-91670406e5bd
# ╠═c3d13c29-569c-4938-9c6a-5a538fe57d3a
# ╠═4be1ed23-02e8-4adc-8699-383db459a295
# ╠═a0a34df2-0653-4a3b-8f69-aa15a959b4ad
# ╠═36f403b0-f576-4868-9b5d-8855bf3e4c90
# ╠═9de2c271-4e4c-4287-8c7f-e9643799db33
# ╟─f9479004-8352-4a99-8e55-37e63d56b898
# ╟─738b39e7-d429-458c-b843-2de6358046f2
# ╟─41883cf3-e8f9-4192-9d65-0b3410d96e0b
# ╠═6bc31402-4be2-4810-aacd-c3a6fd53d69f
# ╠═15934c79-9379-41e3-94c2-76275779e560
# ╠═20057c27-6120-4a56-a316-b29dfcb776a4
# ╠═a22e901e-d71f-47f9-89b3-69c786c977e1
# ╟─5640eb69-ac29-4de0-b3e8-5ba504bda87f
# ╠═5b2b4a34-e4c4-41d4-8720-2afa9480765f
# ╠═eda5fb91-5669-4a31-bfce-0335842f6df6
# ╠═77e159d5-1f79-4e04-b067-b53eb2583ed7
# ╟─1d6bb089-6f64-4777-9922-be3ff8a01af8
# ╠═6b9b0b30-d17a-4cd2-9422-65345f4aecc6
# ╠═3ae9822c-58a7-4e6f-a39d-6d9e88e18d2f
# ╠═23a37e56-a44b-4fec-b90b-5db40aa582aa
# ╠═3431044c-8166-44a7-a6fc-cdd35a72e79b
# ╠═b996450c-0407-48ae-982e-782c1fc796da
# ╠═1635a73a-c522-4796-a8d4-322668f1ab19
# ╠═ca1e599d-b173-478f-9fb7-6ce080f0ee19
# ╠═586e952a-e3f0-46b9-b239-8d1d7a785835
# ╠═01eb6f8d-2787-47ff-b6c8-ccc4e7ebd3e4
# ╠═1c99b55f-d6b4-45af-bdd4-468423f33c8d
# ╠═a19d0448-baee-4a28-8f9f-3c20028e62d0
# ╠═7baae015-6314-4ee2-b716-cc24273a7c39
# ╠═9e9f39b8-2202-46ff-bab7-666f803caf7f
# ╟─1cd26f66-cbf9-4006-8ae6-4851adb8c08d
# ╠═7a48a728-9bad-4838-93a4-7c7ecedf4521
# ╟─f47372b8-d515-4f65-93e6-7a8eaa5d5902
# ╠═c5521565-8e7a-4204-896c-cefb9685eb7d
# ╠═7853c94f-60ff-4e09-98ad-8e6468c73cf0
# ╠═cae4dfc3-0b1c-4cd2-94d9-02dc1fe73553
# ╟─0f241ad4-7b04-4999-bd01-d67898d474d4
# ╟─eec3e5b2-ec1c-45c2-b84f-6fd56fe6f73c
# ╠═f990ad86-7ad9-4f19-beb3-7fed28765166
# ╠═b004f4d1-c28c-4df5-941f-e66d190fb37e
