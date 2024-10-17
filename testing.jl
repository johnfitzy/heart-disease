### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 040ea88d-880f-4937-997e-a15d6a3a4fdc
begin
using Pkg
Pkg.activate(".") 
end

# ╔═╡ 3eea81c2-8c51-11ef-3910-415e5db569ed
using DataFrames, Plots, MLJ, MLJModels, HypothesisTests, CSV, Printf, Plots, MLJLinearModels, StatsPlots, Statistics, CategoricalArrays, Random, DecisionTree,MLJDecisionTreeInterface 

# ╔═╡ d65961b4-2c37-471d-b8fb-254d2c15e400


# ╔═╡ 705511fc-41c5-4043-9ae0-0941c831e908
	restored_mach = MLJ.machine("models/RF_Classifier_DS1_Model.jls")


# ╔═╡ Cell order:
# ╠═040ea88d-880f-4937-997e-a15d6a3a4fdc
# ╠═3eea81c2-8c51-11ef-3910-415e5db569ed
# ╠═d65961b4-2c37-471d-b8fb-254d2c15e400
# ╠═705511fc-41c5-4043-9ae0-0941c831e908
