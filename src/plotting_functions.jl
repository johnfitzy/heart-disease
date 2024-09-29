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
    ; data::Vector{<:Number}, 
    title_text::String,
    xlabel_text::String, 
    ylabel_text::String, 
    labels::Vector{String},
    plot_width::Int = 800,  # Optional width
    plot_height::Int = 400  # Optional height
)
    # Count occurrences of each unique value in the data
    counts = countmap(data)

    # Extract counts for the first two unique values (assumed binary: 0 and 1)
    values = [get(counts, v, 0) for v in [1.0, 0.0]]

    total_count = sum(values)

    # Check to prevent division by zero
    if total_count == 0
        error("Total count is zero, cannot compute percentages.")
    end

    # Calculate percentages
    percentages = [100 * value / total_count for value in values]
    
    # Plotting the Bar Chart with calculated percentages and custom labels
    plt = bar(
        labels, 
        percentages, 
        title=title_text, 
        xlabel=xlabel_text, 
        ylabel=ylabel_text, 
        legend=false,
        yticks=0:10:100,   # Y-axis ticks for percentages
        color=rand(distinguishable_colors(50), 2),  # Random colors for the bars
        size=(plot_width, plot_height)  # Set the plot size
    )

    # Annotate each bar with its percentage value inside the bar
    for (i, percentage) in enumerate(percentages)
        # Adjust the x and y positions for the annotation inside the bar
        x_pos = i - 0.5  # Move the text slightly left (adjust as needed)
        y_pos = percentage / 2  # Place the annotation in the middle of the bar (vertical position)

        annotate!(
            plt, 
            x_pos,  # X position adjustment
            y_pos,  # Y position
            text("$(round(percentage, digits=1))%", :center, 12, :white)  # White text
        )
    end

    plt
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

    # Annotate each bar with its percentage value inside the bar
    for (i, percentage) in enumerate(percentages)
        # Adjust the x and y positions for the annotation inside the bar
        x_pos = i - 0.5  # Move the text slightly left (adjust as needed)
        y_pos = values[i] / 2  # Place the annotation in the middle of the bar (vertical position)

        annotate!(
            plt, 
            x_pos,  # X position adjustment
            y_pos,  # Y position (middle of the bar)
            text("$(round(percentage, digits=1))%", :center, 12, :white)  # White text
        )
    end

    return plt
end

