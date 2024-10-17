### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

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

df_lng_beach = CSV.read("data/DS2_imputed/imputed.processed.long_beach.csv", DataFrame)

df_swiss = CSV.read("data/DS2_imputed/imputed.processed.swiss.csv", DataFrame)
	
df_all = CSV.read("data/DS2_imputed/combined_datasets.csv", DataFrame) 

end

# ╔═╡ b95ef578-11b3-48f5-ad3d-f8a29262f588
# Load up models
begin


## --- Load Logistic Regression Machines --- ##
lg_ds1_mach = machine("models/Logistic_Classifier_DS1_Model.jls")

lg_clev_ds2_mach = machine("models/Logistic_Classifier_DS2_Cleveland_Model.jls")

lg_hung_ds2_mach = machine("models/Logistic_Classifier_DS2_Hungarian_Model.jls")

lg_lb_ds2_mach = machine("models/Logistic_Classifier_DS2_Long_Beach_Model.jls")

lg_swiss_ds2_mach = machine("models/Logistic_Classifier_DS2_Swiss_Model.jls")

	

## --- Load Random Forest Ensemble Classifier Machines --- ##
rf_ensemble_ds1_mach = machine("models/RF_Ensemble_DS1_Model.jls")
	
rf_ensemble_ds2_clev_mach = machine("models/RF_Ensemble_DS2_Cleveland_model.jls")

rf_ensemble_ds2_hung_mach = machine("models/RF_Ensemble_DS2_Hungarian_model.jls")

rf_ensemble_ds2_lb_mach = machine("models/RF_Ensemble_DS2_Longbeach_Model.jls")

rf_ensemble_ds2_swiss_mach = machine("models/RF_Ensemble_DS2_Swiss_model.jls")

## --- Load XGBoost Classifier Machines --- ##
xgboost_ds1_mach = machine("models/XGBoost_Classifier_DS1_Model.jls")

xgboost_ds2_clev_mach = machine("models/XGBoost_Classifier_DS2_Cleveland_Model.jls")

xgboost_ds2_hung_mach = machine("models/XGBoost_Classifer_DS2_Hungarian.jl")

xgboost_ds2_lb_mach = machine("models/XGBoost_Classifier_DS2_LongBeach_Model.jls")

xgboost_ds2_swiss_mach = machine("models/XGBoost_Classifier_DS2_Swiss_Model.jls")

end

# ╔═╡ e920d7ba-911e-47cd-8457-417eee2ebda0
begin
	
# Create a dictionary to map names to DataFrames
lg_files_dict = Dict(
"DS1 Heart Dataset - heart.csv" => df_ds1,
"DS2 Imputed Hungarian Dataset - imputed.processed.hungarian.csv" => df_hung,
"DS2 Imputed Cleveland Dataset - imputed.processed.cleveland.csv" => df_clev,
"DS2 Imputed Long Beach Dataset - imputed.processed.long_beach.csv" => df_lng_beach,	
	"DS2 Imputed Switzerland Dataset - imputed.processed.swiss.csv" => df_swiss,
    "Combined Datasets - combined_datasets.csv" => df_all
)

# Create a dictionary to map names to models
lg_machine_dict = Dict(
    "Model trained on DS1" => lg_ds1_mach,
	"Model trained on DS2 Cleveland" => lg_clev_ds2_mach,
    "Model trained on DS2 Hungarian" => lg_hung_ds2_mach,
	"Model trained on DS2 Long Beach" => lg_lb_ds2_mach,
	"Model trained on DS2 Switzerland" => lg_swiss_ds2_mach
)

end

# ╔═╡ d1a09c15-3d41-4313-823b-76c4f39473d8
lg_machine_name = @bind lg_selected_machine_name Select(collect(keys(lg_machine_dict)))

# ╔═╡ c0a5ec18-e9fe-4f25-b4ce-092470c3d197
lg_file_name = @bind lg_selected_file_name Select(collect(keys(lg_files_dict)))

# ╔═╡ 76f27e4b-198a-4f07-b4da-1e3857c2e7ea
begin
	lg_df = lg_files_dict[lg_selected_file_name]
	lg_mach = lg_machine_dict[lg_selected_machine_name]
end

# ╔═╡ f8159ada-3762-4177-8858-f905ea36b6ff
md"""
## Logistic Regression
"""

# ╔═╡ 7c780262-31c1-4176-8ecc-a9abf9ca0bc4
# Running the Logistic Regression Machine
begin

	# function run_model(df, mach) 
	X, y = coerce_features_and_target_to_scitypes(lg_df)

	# Probabilities generated
	lg_y_prob = MLJ.predict(lg_mach, X)

	# Predictions/classes 
	lg_y_pred = MLJ.predict_mode(lg_mach, X)
end

# ╔═╡ 64d2a229-dad0-4092-bd48-bd5db34e06c7
# Extract metric for later use
begin 
	lg_cm = ConfusionMatrix()(lg_y_pred, y)

	lg_matrix_values = ConfusionMatrices.matrix(lg_cm)

	# Extract TP, FP, TN, FN
	lg_TP = lg_matrix_values[1, 1]  # True Positives
	lg_FP = lg_matrix_values[2, 1]  # False Positives
	lg_TN = lg_matrix_values[2, 2]  # True Negatives
	lg_FN = lg_matrix_values[1, 2]  # False Negatives
	
	lg_accuracy = "$(round(MLJ.accuracy(lg_y_pred, y), sigdigits=4))"
	lg_auc = "$(round(MLJ.auc(lg_y_prob, y), sigdigits=4))"
end

# ╔═╡ 252f0baf-af32-4296-bf11-153aed4b069b
begin
	lg_fprs, lg_tprs, lg_thresholds = roc_curve(lg_y_prob, y)

	lg_roc_plot = plot(lg_fprs, lg_tprs, 
		label="Logistic Classifier", 
		xlabel="False Positive Rate", 
		ylabel="True Positive Rate"
	)

end

# ╔═╡ 8f37976b-4951-4e02-b5e3-f19f8d082c1f
md"""
## Random Forest Ensemble
"""

# ╔═╡ 11ff1cc7-12c4-45bb-996a-a368d9750d01
begin 

	# Probabilities generated
	# y_prob = MLJ.predict(mach, X)

	# # Predictions/classes 
	# y_pred = MLJ.predict_mode(mach, X)




end

# ╔═╡ 9cc84512-c258-48d4-975c-1359380d497b
# Html here

begin

title_html = @htl("""
<h1>Dashboard</h1><br>
<h2>1) Compare Logistic Regression Model Performance Across Datasets</h2><br>
""")

lg_model_select_html = @htl("""
	<h3>Model and Dataset Selection</h3><br>
	<div style="display: flex; flex-direction: column; gap: 10px;">
    	<div>
    <label for="model-select" style="margin-right: 10px;">Select Model:</label>
        $lg_machine_name
    </div>
    <div>
        <label for="file-select" style="margin-right: 10px;">Select File:</label>
        $lg_file_name
    </div>
</div><br>
""")
	
# Manually create the confusion matrix table in HTML
lg_confusion_matrix_html = @htl("""
<table border="1">
  <tr>
    <th> </th><th>Predicted 0</th><th>Predicted 1</th>
  </tr>
  <tr>
    <td>Actual 0</td><td>$lg_TP</td><td>$lg_FN</td>
  </tr>
  <tr>
    <td>Actual 1</td><td>$lg_FP</td><td>$lg_TN</td>
  </tr>
</table>
""")

# Combine everything into an HTML block
lg_roc_out_html = @htl("""
<div>
    <h3>Confusion Matrix</h3>
    $lg_confusion_matrix_html  <!-- Embed the table here -->
    <ul>
        <li>Model accuracy: $lg_accuracy</li>
        <li>AUC: $lg_auc</li>
    </ul>
</div><br>
""")
	
# Wrap the plot in some HTML for display
lg_roc_html = @htl("""
    <div>
        <h3>ROC Curve Analysis</h3><br>
        <p><strong>Selected Model:</strong> $lg_selected_machine_name</p>
        <p><strong>Selected Dataset:</strong> $lg_selected_file_name</p>
        <div>
            $lg_roc_plot
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
    $lg_model_select_html
    $lg_roc_out_html
    $lg_roc_html
	$subtitle_xboost_html
</div>
""")


end

# ╔═╡ 9a9ffaf6-43ca-4eda-9b31-7fc658dd4f9a
# Export HTML cell
begin
	
	@info PlutoRunner.currently_running_cell_id  # get cellid

	layout = vbox([final_html])
end

# ╔═╡ cdc8655f-cb7e-4ecb-bac7-a8cab8a85748
notebook = PlutoRunner.notebook_id[] |> string

# ╔═╡ 7600a7f7-ecbc-4ac5-be5b-b1c7ece70446
celllist=["9a9ffaf6-43ca-4eda-9b31-7fc658dd4f9a"]

# ╔═╡ f3ef1c9e-241d-42fc-84e4-0645ec85cb1f
dash_final_url="http://localhost:1234/edit?" * "id=$notebook&" * join(["isolated_cell_id=$cell" for cell in celllist], "&")

# ╔═╡ 7c6db7cf-795c-43c1-bd7a-2f4f60c850e4
@htl("""
<a href="$dash_final_url" style="font_size=20">Click here for the Dashboard</a>
""")

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
# ╠═8f37976b-4951-4e02-b5e3-f19f8d082c1f
# ╠═11ff1cc7-12c4-45bb-996a-a368d9750d01
# ╠═9cc84512-c258-48d4-975c-1359380d497b
# ╠═9a9ffaf6-43ca-4eda-9b31-7fc658dd4f9a
# ╠═cdc8655f-cb7e-4ecb-bac7-a8cab8a85748
# ╠═7600a7f7-ecbc-4ac5-be5b-b1c7ece70446
# ╠═f3ef1c9e-241d-42fc-84e4-0645ec85cb1f
# ╠═7c6db7cf-795c-43c1-bd7a-2f4f60c850e4
