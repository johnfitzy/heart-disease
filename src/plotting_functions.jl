using Plots, StatsPlots, Distributions

# Define the function for creating a QQ plot comparing data to a normal distribution
function create_qqplot(
    data::Vector{<:Number}, 
    plot_title::String
)
    # Create a QQ plot comparing data to a normal distribution
    qqplot(
        data,
        Normal(mean(data), std(data)),
        title=plot_title, 
        xlabel="Theoretical Quantiles", 
        ylabel="Sample Quantiles"
    )
end


# Define the reusable function for creating a combined violin, dot, and box plot
function create_combined_plot(
    data::Vector{<:Number}, 
    title_text::String, 
    x_label::String, 
    y_label::String
)
    # Create the violin plot
    violin([data], legend=false, xticks=:none)

    # Overlay the dot plot
    dotplot!(data, legend=false, xticks=:none)

    # Overlay the box plot
    boxplot!(data, legend=false, xticks=:none)  # Removes x-axis values

    # Set the title and axis labels
    title!(title_text)
    xaxis!(x_label)
    yaxis!(y_label)
end



# Define the reusable function for creating a combined histogram and density plot
function create_hist_density_plot(
    data::Vector{<:Number}, 
    hist_title::String, 
    density_title::String, 
    hist_xlabel::String, 
    hist_ylabel::String, 
    density_xlabel::String, 
    density_ylabel::String, 
    bins::Int = 10
)
    # Create a plot layout with 1 row and 2 columns
    plt = plot(layout=(1, 2), size=(800, 400))

    # Add a histogram to the first subplot
    histogram!(
        plt[1, 1], 
        data, 
        title=hist_title, 
        xlabel=hist_xlabel, 
        ylabel=hist_ylabel, 
        bins=bins, 
        legend=false
    )

    # Add a density plot to the second subplot
    density!(
        plt[1, 2], 
        data, 
        title=density_title, 
        xlabel=density_xlabel, 
        ylabel=density_ylabel, 
        legend=false
    )

    # Return the combined layout with the two subplots
    return plt
end


# Define the reusable function for creating a bar chart with percentages
function create_binary_bar_plot(
    data::Vector{<:Number}, 
    title_text::String,
    xlabel_text::String, 
    ylabel_text::String, 
    labels::Vector{String}
)
    # Count occurrences of each unique value in the data
    value_counts = countmap(data)

    # Extract counts for the first two unique values (assumed binary: 0 and 1 or similar)
    first_count = get(value_counts, 1.0, 0)
    second_count = get(value_counts, 0.0, 0)

    # Calculate total for percentages
    total = first_count + second_count

    # Check to prevent division by zero
    if total == 0
        error("Total count is zero, cannot compute percentages.")
    end

    # Calculate percentages
    first_percentage = 100 * first_count / total
    second_percentage = 100 * second_count / total

    # Prepare percentages and custom labels with values included
    percentages = [first_percentage, second_percentage]
    custom_labels = [
        "$(labels[1]) $(round(first_percentage, digits=1))%", 
        "$(labels[2]) $(round(second_percentage, digits=1))%"
    ]

    # Plotting the Bar Chart with calculated percentages and custom labels
    bar(
        labels, 
        percentages, 
        title=title_text, 
        xlabel=xlabel_text, 
        ylabel=ylabel_text, 
        label=custom_labels,
        legend=:topright,   # Position the legend to the top right of the graph
        yticks=0:10:100
    )
end


function create_bar_plot(
    ; data::Vector{<:Number}, 
    labels::Vector{String}, 
    values_to_extract::Vector{<:Number}, 
    plot_title::String, 
    xlabel_text::String, 
    ylabel_text::String
)
    # Count occurrences of each category in the data
    counts = countmap(data)

    # Extract counts for the specified values
    values = [get(counts, v, 0) for v in values_to_extract]

    # Calculate total count for percentage calculations
    total_count = sum(values)

    # Calculate percentages
    percentages = [100 * value / total_count for value in values]

    # Plotting the Bar Chart
    plt = bar(
        labels, 
        values, 
        title = plot_title, 
        xlabel = xlabel_text, 
        ylabel = ylabel_text, 
        legend = false,
        bar_width = 0.6, 
        color = rand(distinguishable_colors(50), length(labels)), 
        size = (800, 400)
    )

    # Annotate each bar with its percentage value
    for (i, percentage) in enumerate(percentages)
        annotate!(plt, i, values[i], text("$(round(percentage, digits = 1))%", :right, 8, :black))
    end

    return plt
end

