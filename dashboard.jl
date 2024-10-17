### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 5628a140-b0bf-4bf9-b042-f753c45967d9
begin
using Pkg
Pkg.activate(".") 
end

# ╔═╡ 1eaa54c5-c260-4109-af80-730de1c9e644
using DataFrames, CSV, JLSO, MLJ, MLJLinearModels, Plots, PlutoUI, HypertextLiteral, PrettyTables, MLJDecisionTreeInterface, MLJModels, HypothesisTests, Printf, StatsPlots, Statistics, CategoricalArrays, Random, DecisionTree, MLJMultivariateStatsInterface, MLJXGBoostInterface


# ╔═╡ be908d77-a866-49f7-9fa2-3311d005fa86
using  PlutoUI.ExperimentalLayout: hbox, vbox

# ╔═╡ c250530f-e51f-47c4-ae49-c77cfa756bc8
include("src/data_clean.jl")

# ╔═╡ 3b9392a8-da97-4417-8fc5-6c4eb8c1f090
# Load files into Dataframes
begin

df_ds1 = CSV.read("data/DS1/heart.csv", DataFrame) 

df_hung = CSV.read("data/DS2_imputed/imputed.processed.hungarian.csv", DataFrame) 
	
df_clev = CSV.read("data/DS2_imputed/imputed.processed.cleveland.csv", DataFrame)

df_lb = CSV.read("data/DS2_imputed/imputed.processed.long_beach.csv", DataFrame)

df_swiss = CSV.read("data/DS2_imputed/imputed.processed.swiss.csv", DataFrame)
	
df_all = CSV.read("data/DS2_imputed/combined_datasets.csv", DataFrame) 

end

# ╔═╡ b95ef578-11b3-48f5-ad3d-f8a29262f588
# Load up models
begin


# Model trained on DS1
# lg_ds1_mach = machine("models/Logistic_Classifier_DS1_Model.jls")

# # Model trained on DS2 imputed Hungarian dataset
# lg_clev_mach = machine("models/Logistic_Classifier_DS2_Cleveland_Model.jls")

# lg_hungarian_mach = machine("models/Logistic_Classifier_DS2_Hungarian_Model.jls")

# lg_lb_mach = machine("models/Logistic_Classifier_DS2_Cleveland_Model.jls")

lg_swiss_mach = machine("models/Logistic_Classifier_DS2_Swiss_Model.jls")

	
# Model trained on DS1 with Random Forest
rf_ds1_mach = machine("models/RF_Classifier_DS1_Model.jls")
	

# # Random Forest Ensemble model trained on DS2 Cleveland dataset
# rf_ensemble_ds2_cleveland_mach = machine("models/RF_Ensemble_DS2_Cleveland_model")

# # Random Forest Ensemble model trained on DS2 Hungarian dataset
# rf_ensemble_ds2_hungarian_mach = machine("models/RF_Ensemble_DS2_Hungarian_model")

# # XGBoost model trained on DS1
# xgboost_ds1_mach = machine("models/XGBoost_Classifier_DS1_Model.jls")

# # XGBoost model trained on DS2 Cleveland dataset
# xgboost_ds2_cleveland_mach = machine("models/XGBoost_DS2_Cleveland_Model")

# # XGBoost model trained on DS2 Hungarian dataset
# xgboost_ds2_hungarian_mach = machine("models/XGBoost_DS2_Hungarian_Model")

# # XGBoost model trained on DS2 Long Beach dataset
# xgboost_ds2_longbeach_mach = machine("models/XGBoost_DS2_LongBeach_Model")

# # XGBoost model trained on DS2 Swiss dataset
# xgboost_ds2_swiss_mach = machine("models/XGBoost_DS2_Swiss_Model")


	

end

# ╔═╡ e920d7ba-911e-47cd-8457-417eee2ebda0
# ╠═╡ disabled = true
#=╠═╡
begin
	
# Create a dictionary to map names to DataFrames
files_dict = Dict(
    "DS1 Heart Dataset - heart.csv" => df_ds1,
    "DS2 Imputed Hungarian Dataset - imputed.processed.hungarian.csv" => df_hung,
	"DS2 Imputed Cleveland Dataset - imputed.processed.cleveland.csv" => df_clev,
	"DS2 Imputed Long Beach Dataset - imputed.processed.long_beach.csv" => df_lb,	"DS2 Imputed Switzerland Dataset - imputed.processed.swiss.csv" => df_swiss,
    "Combined Datasets - combined_datasets.csv" => df_all
)

# Create a dictionary to map names to models
machine_dict = Dict(
    "Model trained on DS1" => lg_ds1_mach,
    "Model trained on Hungarian DS2" => lg_hungarian_mach
)

end
  ╠═╡ =#

# ╔═╡ d1a09c15-3d41-4313-823b-76c4f39473d8
# ╠═╡ disabled = true
#=╠═╡
machine_name = @bind selected_machine_name Select(collect(keys(machine_dict)))
  ╠═╡ =#

# ╔═╡ c0a5ec18-e9fe-4f25-b4ce-092470c3d197
# ╠═╡ disabled = true
#=╠═╡
file_name = @bind selected_file_name Select(collect(keys(files_dict)))
  ╠═╡ =#

# ╔═╡ 76f27e4b-198a-4f07-b4da-1e3857c2e7ea
# ╠═╡ disabled = true
#=╠═╡
begin
	df = files_dict[selected_file_name]
	mach = machine_dict[selected_machine_name]
end
  ╠═╡ =#

# ╔═╡ f8159ada-3762-4177-8858-f905ea36b6ff
# ╠═╡ disabled = true
#=╠═╡
md"""
## Logistic Regression
"""
  ╠═╡ =#

# ╔═╡ 7c780262-31c1-4176-8ecc-a9abf9ca0bc4
# ╠═╡ disabled = true
#=╠═╡
# Running the Logistic Regression Machine
begin

	# function run_model(df, mach) 
	X, y = coerce_features_and_target_to_scitypes(df)

	# Probabilities generated
	y_prob = predict(mach, X)

	# Predictions/classes 
	y_pred = predict_mode(mach, X)
end
  ╠═╡ =#

# ╔═╡ 64d2a229-dad0-4092-bd48-bd5db34e06c7
# ╠═╡ disabled = true
#=╠═╡
# Extract metric for later use
begin 
	cm = ConfusionMatrix()(y_pred, y)

	matrix_values = ConfusionMatrices.matrix(cm)

	# Extract TP, FP, TN, FN
	TP = matrix_values[1, 1]  # True Positives
	FP = matrix_values[2, 1]  # False Positives
	TN = matrix_values[2, 2]  # True Negatives
	FN = matrix_values[1, 2]  # False Negatives
	
	accuracy = "$(round(MLJ.accuracy(y_pred, y), sigdigits=4))"
	auc = "$(round(MLJ.auc(y_prob, y), sigdigits=4))"
end
  ╠═╡ =#

# ╔═╡ 252f0baf-af32-4296-bf11-153aed4b069b
# ╠═╡ disabled = true
#=╠═╡
begin
	fprs, tprs, thresholds = roc_curve(y_prob, y)

	roc_plot = plot(fprs, tprs, 
		label="Logistic Classifier", 
		xlabel="False Positive Rate", 
		ylabel="True Positive Rate"
	)

end
  ╠═╡ =#

# ╔═╡ 8f37976b-4951-4e02-b5e3-f19f8d082c1f
md"""
## XBoost
"""

# ╔═╡ 9cc84512-c258-48d4-975c-1359380d497b
# ╠═╡ disabled = true
#=╠═╡
# Html here

begin

title_html = @htl("""
<h1>Dashboard</h1><br>
<h2>1) Compare Logistic Regression Model Performance Across Datasets</h2><br>
""")

log_model_select_html = @htl("""
	<h3>Model and Dataset Selection</h3><br>
	<div style="display: flex; flex-direction: column; gap: 10px;">
    	<div>
    <label for="model-select" style="margin-right: 10px;">Select Model:</label>
        $machine_name
    </div>
    <div>
        <label for="file-select" style="margin-right: 10px;">Select File:</label>
        $file_name
    </div>
</div><br>
""")
	
# Manually create the confusion matrix table in HTML
log_confusion_matrix_html = @htl("""
<table border="1">
  <tr>
    <th> </th><th>Predicted 0</th><th>Predicted 1</th>
  </tr>
  <tr>
    <td>Actual 0</td><td>$TP</td><td>$FN</td>
  </tr>
  <tr>
    <td>Actual 1</td><td>$FP</td><td>$TN</td>
  </tr>
</table>
""")

# Combine everything into an HTML block
log_roc_out_html = @htl("""
<div>
    <h3>Confusion Matrix</h3>
    $log_confusion_matrix_html  <!-- Embed the table here -->
    <ul>
        <li>Model accuracy: $accuracy</li>
        <li>AUC: $auc</li>
    </ul>
</div><br>
""")
	
# Wrap the plot in some HTML for display
roc_html = @htl("""
    <div>
        <h3>ROC Curve Analysis</h3><br>
        <p><strong>Selected Model:</strong> $selected_machine_name</p>
        <p><strong>Selected Dataset:</strong> $selected_file_name</p>
        <div>
            $roc_plot
        </div>
    </div>
""")

subtitle_xboost_html = @htl("""
<h2>2) Compare Gradient Boosting -  XGBoost Classifier Model's Performance Across Datasets</h2><br>
""")

# Final Layout with all HTML elements combined into one big block
final_html = @htl("""
<div style="max-width: 800px; margin: auto;">
    $title_html
    $log_model_select_html
    $log_roc_out_html
    $roc_html
	$subtitle_xboost_html
</div>
""")


end
  ╠═╡ =#

# ╔═╡ 9a9ffaf6-43ca-4eda-9b31-7fc658dd4f9a
# ╠═╡ disabled = true
#=╠═╡
# Export HTML cell
begin
	
	@info PlutoRunner.currently_running_cell_id  # get cellid

	layout = vbox([final_html])
end
  ╠═╡ =#

# ╔═╡ cdc8655f-cb7e-4ecb-bac7-a8cab8a85748
# ╠═╡ disabled = true
#=╠═╡
notebook = PlutoRunner.notebook_id[] |> string
  ╠═╡ =#

# ╔═╡ 7600a7f7-ecbc-4ac5-be5b-b1c7ece70446
# ╠═╡ disabled = true
#=╠═╡
celllist=["9a9ffaf6-43ca-4eda-9b31-7fc658dd4f9a"]
  ╠═╡ =#

# ╔═╡ f3ef1c9e-241d-42fc-84e4-0645ec85cb1f
# ╠═╡ disabled = true
#=╠═╡
dash_final_url="http://localhost:1234/edit?" * "id=$notebook&" * join(["isolated_cell_id=$cell" for cell in celllist], "&")
  ╠═╡ =#

# ╔═╡ 7c6db7cf-795c-43c1-bd7a-2f4f60c850e4
# ╠═╡ disabled = true
#=╠═╡
@htl("""
<a href="$dash_final_url" style="font_size=20">Click here for the Dashboard</a>
""")
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═5628a140-b0bf-4bf9-b042-f753c45967d9
# ╠═1eaa54c5-c260-4109-af80-730de1c9e644
# ╠═be908d77-a866-49f7-9fa2-3311d005fa86
# ╠═c250530f-e51f-47c4-ae49-c77cfa756bc8
# ╠═3b9392a8-da97-4417-8fc5-6c4eb8c1f090
# ╠═b95ef578-11b3-48f5-ad3d-f8a29262f588
# ╠═e920d7ba-911e-47cd-8457-417eee2ebda0
# ╠═d1a09c15-3d41-4313-823b-76c4f39473d8
# ╠═c0a5ec18-e9fe-4f25-b4ce-092470c3d197
# ╠═76f27e4b-198a-4f07-b4da-1e3857c2e7ea
# ╠═f8159ada-3762-4177-8858-f905ea36b6ff
# ╠═7c780262-31c1-4176-8ecc-a9abf9ca0bc4
# ╠═64d2a229-dad0-4092-bd48-bd5db34e06c7
# ╠═252f0baf-af32-4296-bf11-153aed4b069b
# ╟─8f37976b-4951-4e02-b5e3-f19f8d082c1f
# ╠═9cc84512-c258-48d4-975c-1359380d497b
# ╠═9a9ffaf6-43ca-4eda-9b31-7fc658dd4f9a
# ╠═cdc8655f-cb7e-4ecb-bac7-a8cab8a85748
# ╠═7600a7f7-ecbc-4ac5-be5b-b1c7ece70446
# ╠═f3ef1c9e-241d-42fc-84e4-0645ec85cb1f
# ╠═7c6db7cf-795c-43c1-bd7a-2f4f60c850e4
