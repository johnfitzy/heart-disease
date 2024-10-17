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
files_dict = Dict(
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

# Create a dictionary to map names to Random Forest (RF) models
rf_machine_dict = Dict(
    "Random Forest Ensemble Model trained on DS1" => rf_ensemble_ds1_mach,
    "Random Forest Ensemble Model trained on DS2 Cleveland" => rf_ensemble_ds2_clev_mach,
    "Random Forest Ensemble Model trained on DS2 Hungarian" => rf_ensemble_ds2_hung_mach,
    "Random Forest Ensemble Model trained on DS2 Long Beach" => rf_ensemble_ds2_lb_mach,
    "Random Forest Ensemble Model trained on DS2 Switzerland" => rf_ensemble_ds2_swiss_mach
)

# Create a dictionary to map names to XGBoost models
xgboost_mach_dict = Dict(
    "XGBoost Classifier Model trained on DS1" => xgboost_ds1_mach,
    "XGBoost Classifier Model trained on DS2 Cleveland" => xgboost_ds2_clev_mach,
    "XGBoost Classifier Model trained on DS2 Hungarian" => xgboost_ds2_hung_mach,
    "XGBoost Classifier Model trained on DS2 Long Beach" => xgboost_ds2_lb_mach,
    "XGBoost Classifier Model trained on DS2 Switzerland" => xgboost_ds2_swiss_mach
)


end

# ╔═╡ 8ce3e9e2-1c27-4e22-8372-f49c308896d9
begin
function calculate_precision(TP, FP)
    if TP + FP == 0
        return "Undefined (No positive predictions)"
    else
        precision = TP / (TP + FP)
        return precision
    end
end
metrics = ["Accuracy", "Precision"]
end

# ╔═╡ 8402081d-85a7-4c2e-acdb-b9a33520b9e7
md"""
## Logistic Regression
-------------------
"""

# ╔═╡ d1a09c15-3d41-4313-823b-76c4f39473d8
lg_mach_name = @bind lg_selected_mach_name Select(collect(keys(lg_machine_dict)))

# ╔═╡ c0a5ec18-e9fe-4f25-b4ce-092470c3d197
lg_file_name = @bind lg_selected_file_name Select(collect(keys(files_dict)))

# ╔═╡ 76f27e4b-198a-4f07-b4da-1e3857c2e7ea
begin
	lg_df = files_dict[lg_selected_file_name]
	lg_mach = lg_machine_dict[lg_selected_mach_name]
end

# ╔═╡ 7c780262-31c1-4176-8ecc-a9abf9ca0bc4
# Running the Logistic Regression Machine
begin

	# function run_model(df, mach) 
	lg_X, lg_y = coerce_features_and_target_to_scitypes(lg_df)

	# Probabilities generated
	lg_y_prob = MLJ.predict(lg_mach, lg_X)

	# Predictions/classes 
	lg_y_pred = MLJ.predict_mode(lg_mach, lg_X)
end

# ╔═╡ 64d2a229-dad0-4092-bd48-bd5db34e06c7
# Extract metric for later use
begin 
	lg_cm = ConfusionMatrix()(lg_y_pred, lg_y)

	lg_matrix_values = ConfusionMatrices.matrix(lg_cm)

	# Extract TP, FP, TN, FN
	lg_TP = lg_matrix_values[1, 1]  # True Positives
	lg_FP = lg_matrix_values[2, 1]  # False Positives
	lg_TN = lg_matrix_values[2, 2]  # True Negatives
	lg_FN = lg_matrix_values[1, 2]  # False Negatives
	
	lg_accuracy = round(MLJ.accuracy(lg_y_pred, lg_y), sigdigits=4)
	lg_auc = round(MLJ.auc(lg_y_prob, lg_y), sigdigits=4)
	lg_precision = round(calculate_precision(lg_TP, lg_FP), sigdigits=4)

end

# ╔═╡ ed15e30b-7f33-430a-b418-ccd89d1ed849
begin

# From https://archive.ics.uci.edu/dataset/45/heart+disease
lg_accuracy_baseline = 81.579
lg_precision_baseline = 83.185

lg_data_model = [lg_accuracy * 100, lg_precision * 100]
lg_data_baseline = [lg_accuracy_baseline, lg_precision_baseline]

lg_dot_plot = scatter(
    metrics,
    lg_data_model,
    markersize = 4,
    label = "Our Model",
    xlabel = "Metrics",
    ylabel = "Scores",
    legend = :bottom, 
	color = :red
)

# Overlay the baseline metrics using scatter!
scatter!(
    metrics,
    lg_data_baseline,
    markersize = 4,
    label = "Baseline",
    color = :green
)

annotate!(0.6, lg_data_model[1], 
	text("$(round(lg_data_model[1], digits=2)) %", 10, :red))
	
annotate!(0.6, lg_data_baseline[1], 
	text("$(round(lg_data_baseline[1], digits=2)) %", 10, :green))


annotate!(1.4, lg_data_model[2], 
	text("$(round(lg_data_model[2], digits=2)) %", 10, :red))

annotate!(1.4, lg_data_baseline[2], 
	text("$(round(lg_data_baseline[2], digits=2)) %", 10, :green))


lg_dot_plot


end

# ╔═╡ 605c1211-30d4-4c8f-928a-e8816fd42d0a
lg_data_model

# ╔═╡ 252f0baf-af32-4296-bf11-153aed4b069b
begin
	lg_fprs, lg_tprs, lg_thresholds = roc_curve(lg_y_prob, lg_y)

	lg_roc_plot = plot(lg_fprs, lg_tprs, 
		label="Logistic Classifier", 
		xlabel="False Positive Rate", 
		ylabel="True Positive Rate"
	)

end

# ╔═╡ 8f37976b-4951-4e02-b5e3-f19f8d082c1f
md"""
## Random Forest Ensemble Classifier
------------------------
"""

# ╔═╡ 50f5f871-42b3-4bf0-bf01-2382d07a19d1
rf_mach_name = @bind rf_selected_mach_name Select(collect(keys(rf_machine_dict)))

# ╔═╡ 54e99ead-caa0-46d3-a137-deba640c5b11
rf_file_name = @bind rf_selected_file_name Select(collect(keys(files_dict)))

# ╔═╡ ba600b4f-d293-420d-8aed-8af2f5a48526
begin
	rf_df = files_dict[rf_selected_file_name]
	rf_mach = rf_machine_dict[rf_selected_mach_name]
end

# ╔═╡ 11ff1cc7-12c4-45bb-996a-a368d9750d01
begin 

	# function run_model(df, mach) 
	rf_X, rf_y = coerce_features_and_target_to_scitypes(rf_df)

	# Probabilities generated
	rf_y_prob = MLJ.predict(rf_mach, rf_X)

	# Predictions/classes 
	rf_y_pred = MLJ.predict_mode(rf_mach, rf_X)

end

# ╔═╡ db3eeb77-a0f3-449b-9acb-d1462a84ea29
# Extract metric for later use
begin 
rf_cm = ConfusionMatrix()(rf_y_pred, rf_y)

rf_matrix_values = ConfusionMatrices.matrix(rf_cm)

# Extract TP, FP, TN, FN
rf_TP = rf_matrix_values[1, 1]  # True Positives
rf_FP = rf_matrix_values[2, 1]  # False Positives
rf_TN = rf_matrix_values[2, 2]  # True Negatives
rf_FN = rf_matrix_values[1, 2]  # False Negatives

rf_accuracy = round(MLJ.accuracy(rf_y_pred, rf_y), sigdigits=4)
rf_auc = round(MLJ.auc(rf_y_prob, rf_y), sigdigits=4)
rf_precision = round(calculate_precision(rf_TP, rf_FP), sigdigits=4)


end

# ╔═╡ bb3bd772-7c21-4ce6-8ba0-9d6daf3db6ba
begin
rf_accuracy_baseline = 80.263
rf_precision_baseline = 82.20

rf_data_model = [rf_accuracy * 100, rf_precision * 100]
rf_data_baseline = [rf_accuracy_baseline, rf_precision_baseline]

rf_dot_plot = scatter(
    metrics,
    rf_data_model,
    markersize = 4,
    label = "Our Model",
    xlabel = "Metrics",
    ylabel = "Scores",
    legend = :bottom, 
	color = :red
)

# Overlay the baseline metrics using scatter!
scatter!(
    metrics,
    rf_data_baseline,
    markersize = 4,
    label = "Baseline",
    color = :green
)

annotate!(0.6, rf_data_model[1], 
	text("$(round(rf_data_model[1], digits=2)) %", 10, :red))
	
annotate!(0.6, rf_data_baseline[1], 
	text("$(round(rf_data_baseline[1], digits=2)) %", 10, :green))


annotate!(1.4, rf_data_model[2], 
	text("$(round(rf_data_model[2], digits=2)) %", 10, :red))

annotate!(1.4, rf_data_baseline[2], 
	text("$(round(rf_data_baseline[2], digits=2)) %", 10, :green))


rf_dot_plot

end

# ╔═╡ b3839eba-b51e-4cbd-9a3c-864ecd239cd1
begin
	
rf_fprs, rf_tprs, rf_thresholds = roc_curve(rf_y_prob, rf_y)

rf_roc_plot = plot(rf_fprs, rf_tprs, 
    label="Random Forest Classifier", 
    xlabel="False Positive Rate", 
    ylabel="True Positive Rate"
)

end

# ╔═╡ 047b34f8-abb0-4888-9860-b539f0493cb4
md"""
## XGBoost Classifier
------------------------
"""

# ╔═╡ f4bd9cb2-cfcd-4000-831d-5778b2014ffc
xg_mach_name = @bind xg_selected_mach_name Select(collect(keys(xgboost_mach_dict)))

# ╔═╡ b3f7a4f5-a907-4f9c-b9ad-caa2cd443a2f
xg_file_name = @bind xg_selected_file_name Select(collect(keys(files_dict)))

# ╔═╡ bc0293ea-fe57-49a6-8cd7-376bafbd357a
begin
xg_df = files_dict[xg_selected_file_name]
xg_mach = xgboost_mach_dict[xg_selected_mach_name]
end

# ╔═╡ 5b501edf-b323-43b4-a02a-0202fc8b3f4d
begin

# function run_model(df, mach) 
xg_X, xg_y = coerce_features_and_target_to_scitypes(xg_df)

# Probabilities generated
xg_y_prob = MLJ.predict(xg_mach, xg_X)

# Predictions/classes 
xg_y_pred = MLJ.predict_mode(xg_mach, xg_X)


end

# ╔═╡ 70c6ee90-1ee8-4dd9-ba95-1d535c674f04
begin
xg_cm = ConfusionMatrix()(xg_y_pred, xg_y)

xg_matrix_values = ConfusionMatrices.matrix(xg_cm)

# Extract TP, FP, TN, FN
xg_TP = xg_matrix_values[1, 1]  # True Positives
xg_FP = xg_matrix_values[2, 1]  # False Positives
xg_TN = xg_matrix_values[2, 2]  # True Negatives
xg_FN = xg_matrix_values[1, 2]  # False Negatives

xg_accuracy = round(MLJ.accuracy(xg_y_pred, xg_y), sigdigits=4)
xg_auc = round(MLJ.auc(xg_y_prob, xg_y), sigdigits=4)
xg_precision = round(calculate_precision(xg_TP, xg_FP), sigdigits=4)

end

# ╔═╡ 57f9d989-06e7-4dd8-b133-e1fb52071d0a
begin

xg_accuracy_baseline = 81.679
xg_precision_baseline = 83.185

xg_data_model = [xg_accuracy * 100, xg_precision * 100]
xg_data_baseline = [xg_accuracy_baseline, xg_precision_baseline]

xg_dot_plot = scatter(
    metrics,
    xg_data_model,
    markersize = 4,
    label = "Our Model",
    xlabel = "Metrics",
    ylabel = "Scores",
    legend = :bottom, 
    color = :red
)

# Overlay the baseline metrics using scatter!
scatter!(
    metrics,
    xg_data_baseline,
    markersize = 4,
    label = "Baseline",
    color = :green
)

annotate!(0.6, xg_data_model[1], 
    text("$(round(xg_data_model[1], digits=2)) %", 10, :red))
    
annotate!(0.6, xg_data_baseline[1], 
    text("$(round(xg_data_baseline[1], digits=2)) %", 10, :green))

annotate!(1.4, xg_data_model[2], 
    text("$(round(xg_data_model[2], digits=2)) %", 10, :red))

annotate!(1.4, xg_data_baseline[2], 
    text("$(round(xg_data_baseline[2], digits=2)) %", 10, :green))

xg_dot_plot

end

# ╔═╡ 7596c18a-1133-4057-948f-1f5bae740a12
begin

xg_fprs, xg_tprs, xg_thresholds = roc_curve(xg_y_prob, xg_y)

xg_roc_plot = plot(xg_fprs, xg_tprs, 
    label="XGBoost Classifier", 
    xlabel="False Positive Rate", 
    ylabel="True Positive Rate"
)
end

# ╔═╡ 9cc84512-c258-48d4-975c-1359380d497b
# Html here

begin


		### --- HTML for Logistic Regression --- ###

title_html = @htl("""
<h1>Dashboard</h1><br>
<h2>1) Compare Logistic Regression Model Performance Across Datasets</h2><br>
""")

lg_model_select_html = @htl("""
	<h3>Model and Dataset Selection</h3><br>
	<div style="display: flex; flex-direction: column; gap: 10px;">
    	<div>
    <label for="model-select" style="margin-right: 10px;">Select Model:</label>
        $lg_mach_name
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
		<li>Precision: $lg_precision</li>
    </ul>
</div><br>
""")

lg_dot_plt_html = @htl("""
    <div>
         <h3>Our Model Metrics vs Baseline</h3>
		<p>Baseline source: <a href="https://archive.ics.uci.edu/dataset/45/heart+disease">heart dataset</a>
		</p>
        <p><strong>Selected Model:</strong> $lg_selected_mach_name</p>
        <p><strong>Selected Dataset:</strong> $lg_selected_file_name</p>
        <div>
            $lg_dot_plot
        </div>
    </div>
""")
	
# Wrap the plot in some HTML for display
lg_roc_plt_html = @htl("""
    <div>
        <h3>ROC Curve Analysis</h3><br>
        <p><strong>Selected Model:</strong> $lg_selected_mach_name</p>
        <p><strong>Selected Dataset:</strong> $lg_selected_file_name</p>
        <div>
            $lg_roc_plot
        </div>
    </div>
""")

	### --- HTML for Random Forest --- ###

rf_subtitle_html = @htl("""
<h2>2) Compare Random Forest Classifier Model's Performance Across Datasets</h2><br>
""")

rf_model_select_html = @htl("""
	<h3>Model and Dataset Selection</h3><br>
	<div style="display: flex; flex-direction: column; gap: 10px;">
    	<div>
    <label for="model-select" style="margin-right: 10px;">Select Model:</label>
        $rf_mach_name
    </div>
    <div>
        <label for="file-select" style="margin-right: 10px;">Select File:</label>
        $rf_file_name
    </div>
</div><br>
""")
	
# Manually create the confusion matrix table in HTML
rf_confusion_matrix_html = @htl("""
<table border="1">
  <tr>
    <th> </th><th>Predicted 0</th><th>Predicted 1</th>
  </tr>
  <tr>
    <td>Actual 0</td><td>$rf_TP</td><td>$rf_FN</td>
  </tr>
  <tr>
    <td>Actual 1</td><td>$rf_FP</td><td>$rf_TN</td>
  </tr>
</table>
""")

# Combine everything into an HTML block
rf_roc_out_html = @htl("""
<div>
    <h3>Confusion Matrix</h3>
    $rf_confusion_matrix_html  <!-- Embed the table here -->
    <ul>
        <li>Model accuracy: $rf_accuracy</li>
        <li>AUC: $rf_auc</li>
		<li>Precision: $rf_precision</li>
    </ul>
</div><br>
""")

rf_dot_plt_html = @htl("""
    <div>
         <h3>Our Model Metrics vs Baseline</h3>
		<p>Baseline source: <a href="https://archive.ics.uci.edu/dataset/45/heart+disease">heart dataset</a>
		</p>
        <p><strong>Selected Model:</strong> $rf_selected_mach_name</p>
        <p><strong>Selected Dataset:</strong> $rf_selected_file_name</p>
        <div>
            $rf_dot_plot
        </div>
    </div>
""")

	
# Wrap the plot in some HTML for display
rf_roc_plt_html = @htl("""
    <div>
        <h3>ROC Curve Analysis</h3><br>
        <p><strong>Selected Model:</strong> $rf_selected_mach_name</p>
        <p><strong>Selected Dataset:</strong> $rf_selected_file_name</p>
        <div>
            $rf_roc_plot
        </div>
    </div>
""")

	## --- HTML for XGBoost --- ##
	
xg_subtitle_html = @htl("""
<h2>3) Compare XGBoost Classifier Model's Performance Across Datasets</h2><br>
""")

xg_model_select_html = @htl("""
    <h3>Model and Dataset Selection</h3><br>
    <div style="display: flex; flex-direction: column; gap: 10px;">
        <div>
    <label for="model-select" style="margin-right: 10px;">Select Model:</label>
        $xg_mach_name
    </div>
    <div>
        <label for="file-select" style="margin-right: 10px;">Select File:</label>
        $xg_file_name
    </div>
</div><br>
""")

# Manually create the confusion matrix table in HTML
xg_confusion_matrix_html = @htl("""
<table border="1">
  <tr>
    <th> </th><th>Predicted 0</th><th>Predicted 1</th>
  </tr>
  <tr>
    <td>Actual 0</td><td>$xg_TP</td><td>$xg_FN</td>
  </tr>
  <tr>
    <td>Actual 1</td><td>$xg_FP</td><td>$xg_TN</td>
  </tr>
</table>
""")

# Combine everything into an HTML block
xg_roc_out_html = @htl("""
<div>
    <h3>Confusion Matrix</h3>
    $xg_confusion_matrix_html  <!-- Embed the table here -->
    <ul>
        <li>Model accuracy: $xg_accuracy</li>
        <li>AUC: $xg_auc</li>
		<li>Precision: $xg_precision</li>
    </ul>
</div><br>
""")

xg_dot_plt_html = @htl("""
    <div>
         <h3>Our Model Metrics vs Baseline</h3>
		<p>Baseline source: <a href="https://archive.ics.uci.edu/dataset/45/heart+disease">heart dataset</a></p>
        <p><strong>Selected Model:</strong> $xg_selected_mach_name</p>
        <p><strong>Selected Dataset:</strong> $xg_selected_file_name</p>
        <div>
            $xg_dot_plot
        </div>
    </div>
""")


# Wrap the plot in some HTML for display
xg_roc_plt_html = @htl("""
    <div>
        <h3>ROC Curve Analysis</h3><br>
        <p><strong>Selected Model:</strong> $xg_selected_mach_name</p>
        <p><strong>Selected Dataset:</strong> $xg_selected_file_name</p>
        <div>
            $xg_roc_plot
        </div>
    </div>
""")

	final_html = @htl("""
<div style="max-width: 800px; margin: auto;">
    $title_html
    $lg_model_select_html
    $lg_roc_out_html
	$lg_dot_plt_html
    $lg_roc_plt_html
	$rf_subtitle_html
    $rf_model_select_html
    $rf_roc_out_html
	$rf_dot_plt_html
    $rf_roc_plt_html
	$xg_subtitle_html
    $xg_model_select_html
    $xg_roc_out_html
	$xg_dot_plt_html
    $xg_roc_plt_html
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
# ╠═8ce3e9e2-1c27-4e22-8372-f49c308896d9
# ╟─8402081d-85a7-4c2e-acdb-b9a33520b9e7
# ╠═d1a09c15-3d41-4313-823b-76c4f39473d8
# ╠═c0a5ec18-e9fe-4f25-b4ce-092470c3d197
# ╠═76f27e4b-198a-4f07-b4da-1e3857c2e7ea
# ╠═7c780262-31c1-4176-8ecc-a9abf9ca0bc4
# ╠═64d2a229-dad0-4092-bd48-bd5db34e06c7
# ╠═ed15e30b-7f33-430a-b418-ccd89d1ed849
# ╠═605c1211-30d4-4c8f-928a-e8816fd42d0a
# ╠═252f0baf-af32-4296-bf11-153aed4b069b
# ╠═8f37976b-4951-4e02-b5e3-f19f8d082c1f
# ╠═50f5f871-42b3-4bf0-bf01-2382d07a19d1
# ╠═54e99ead-caa0-46d3-a137-deba640c5b11
# ╠═ba600b4f-d293-420d-8aed-8af2f5a48526
# ╠═11ff1cc7-12c4-45bb-996a-a368d9750d01
# ╠═db3eeb77-a0f3-449b-9acb-d1462a84ea29
# ╠═bb3bd772-7c21-4ce6-8ba0-9d6daf3db6ba
# ╠═b3839eba-b51e-4cbd-9a3c-864ecd239cd1
# ╠═047b34f8-abb0-4888-9860-b539f0493cb4
# ╠═f4bd9cb2-cfcd-4000-831d-5778b2014ffc
# ╠═b3f7a4f5-a907-4f9c-b9ad-caa2cd443a2f
# ╠═bc0293ea-fe57-49a6-8cd7-376bafbd357a
# ╠═5b501edf-b323-43b4-a02a-0202fc8b3f4d
# ╠═70c6ee90-1ee8-4dd9-ba95-1d535c674f04
# ╠═57f9d989-06e7-4dd8-b133-e1fb52071d0a
# ╠═7596c18a-1133-4057-948f-1f5bae740a12
# ╠═9cc84512-c258-48d4-975c-1359380d497b
# ╠═9a9ffaf6-43ca-4eda-9b31-7fc658dd4f9a
# ╠═cdc8655f-cb7e-4ecb-bac7-a8cab8a85748
# ╠═7600a7f7-ecbc-4ac5-be5b-b1c7ece70446
# ╠═f3ef1c9e-241d-42fc-84e4-0645ec85cb1f
# ╠═7c6db7cf-795c-43c1-bd7a-2f4f60c850e4
