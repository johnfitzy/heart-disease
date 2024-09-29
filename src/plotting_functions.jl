
using Plots, StatsPlots


function create_qqplot(data::Vector{<:Number}, plot_title::String)
	# Create a QQ plot comparing data to a normal distribution
	qqplot(data,
		Normal(mean(data), std(data)),
        	title=plot_title, 
        	xlabel="Theoretical Quantiles", 
        	ylabel="Sample Quantiles"
	)
end
