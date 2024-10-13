"""
    clean_value(x, desired_type)

Cleans and converts the input `x` to the specified `desired_type`. If `x` is a string, it removes leading and trailing whitespace and attempts to parse it as the `desired_type`. If parsing fails, it returns `missing`. If `x` is already of the correct type or is `missing`, it is returned unchanged.

# Arguments
- `x`: The value to be cleaned and converted (can be a string, number, or `missing`).
- `desired_type`: The target type to which `x` should be converted (e.g., `Int64`, `Float64`).

# Example
```julia
clean_value(" 42 ", Int64)  # Returns 42
clean_value("invalid", Int64)  # Returns missing
clean_value(missing, Float64)  # Returns missing
"""

#function clean_value(x, desired_type)
#     if x isa AbstractString
#         stripped_x = strip(x)
#         try
#             return parse(desired_type, stripped_x)
#         catch e
#             return missing
#         end
#     elseif x isa Number
#         return convert(desired_type, x)  # Convert to the desired type if it's a number
#     else
#         return x  # Return as is for missing or already cleaned values
#     end
# end


function clean_value(x, desired_type)
    	if x isa Float64 && desired_type == Int64
		return round(Int, x)
	elseif x isa AbstractString
        stripped_x = strip(x)
	
	
    try
		# as desired
        return parse(desired_type, stripped_x)
    catch e
        if e isa ArgumentError
            try
				# Crappy code - fall back option is float to int
                return round(Int, parse(Float64, stripped_x))
            catch
                # If it fails, return missing
                return missing

            end
        else
            # Re-throw if it's another type of error
            rethrow(e)
        end
    end
    end

	return x

end

"""
    clean_and_flatten_column!(df::DataFrame, col_name::Symbol, desired_type::Type)

Flattens and cleans the specified column `col_name` in the DataFrame `df` by:

1. Flattening the column if it's a `PooledArray`.
2. Converting the values in the column to the `desired_type`, handling missing or invalid data appropriately.

# Arguments
- `df`: A `DataFrame` containing the column to be cleaned.
- `col_name`: The name of the column (as a `Symbol`) to clean and flatten.
- `desired_type`: The type (e.g., `Int64`, `Float64`) to which the column's values will be converted.

# Example
```julia
clean_and_flatten_column!(df, :age, Float64)
"""

function clean_and_flatten_column!(df, col_name, desired_type)
    df[!, col_name] = collect(df[!, col_name])  # Flatten if it's a PooledArray
    df[!, col_name] = map(x -> clean_value(x, desired_type), df[!, col_name])
end

"""
    clean_and_flatten!(df::DataFrame)

Cleans and flattens the columns of the given DataFrame `df` by performing the following steps:

1. Checks that the column names of `df` match a predefined list of expected column names. If not, an error is thrown.
2. Flattens PooledArray columns (if present) and converts each column to its appropriate data type:
   - Columns like `"heart_disease"` are converted to `Int64`.
   - All other columns are converted to `Float64`.

# Arguments
- `df`: A `DataFrame` to be cleaned and processed.

# Throws
- `ErrorException`: If the column names in `df` do not match the expected column names.

# Example
```julia
clean_and_flatten!(df)
"""

function clean_and_flatten!(df)

    expected_column_names = [
        "age", 
        "sex", 
        "chest_pain", 
        "rest_bp", 
        "serum_chol", 
        "fasting_blood_sugar", 
        "electrocardiographic", 
        "max_heart_rate", 
        "angina", 
        "oldpeak", 
        "slope", 
        "major_vessels", 
        "thal", 
        "heart_disease"
    ]

    # Check for column name mismatch
    if Set(names(df)) != Set(expected_column_names)
        throw(ErrorException("Column names do not match the expected names."))
    end

    # Clean and flatten each column based on its type
    for col_name in names(df)
        if col_name in ["sex", "fasting_blood_sugar", "angina", "slope", "heart_disease", "age", "chest_pain", "electrocardiographic", "thal"]
            clean_and_flatten_column!(df, col_name, Int64)  # Binary and Integer types
        else
            clean_and_flatten_column!(df, col_name, Float64)  # All other continuous variables as floats
        end
    end

end

"""
    print_column_types(df::DataFrame)

Prints the data types of all columns in the given DataFrame.

### Arguments
- `df::DataFrame`: The DataFrame whose column types will be printed.

### Example
```julia
using DataFrames

# Example DataFrame
df = DataFrame(
    a = [1, 2, 3],
    b = ["one", "two", "three"],
    c = [1.1, 2.2, 3.3]
)

# Call the function to print column types
print_column_types(df)
# Output:
# Column a: Int64
# Column b: String
# Column c: Float64
"""

function print_column_types(df::DataFrame)
    for col in names(df)
        println(typeof(df[!, col]))
    end
end

"""
    missing_datapoints_percent(df::DataFrame) -> Vector{Float64}

Calculates the percentage of missing data points for each column in the given DataFrame `df`. 

A data point is considered "missing" if it is either `missing` or the string `"?"`. The function returns a vector containing the percentage of missing data points for each column in the DataFrame, where the percentages are based on the total number of rows in each column.

# Arguments
- `df::DataFrame`: The input DataFrame where each column is analyzed for missing data.

# Returns
- A vector of `Float64` values, where each value represents the percentage of missing data points in the corresponding column of the DataFrame.

# Example
```julia
using DataFrames

df = DataFrame(A = [1, 2, missing], B = ["?", "value", "value"])
missing_datapoints_percent(df)
# Output: [33.33, 33.33]
"""

function missing_datapoints_percent(df::DataFrame)
	col_size = size(df)[1]
	return [(count(x -> ismissing(x) || x == "?", df[!, col]) / col_size) * 100 for col in names(df)]
end



"""
    coerce_features_and_target_to_scitypes(df::DataFrame)

Coerces the input DataFrame `df` to the required scientific types for model training.

# Arguments
- `df::DataFrame`: A DataFrame where the first 13 columns represent the features and the 14th column is the target variable.

# Returns
- `X`: A DataFrame where the first 13 feature columns have been coerced to `Continuous` scientific type.
- `y`: A vector where the 14th column (target) has been coerced to `OrderedFactor` scientific type.

# Example
```julia
X, y = coerce_features_and_target_to_scitypes(df)

"""

function coerce_features_and_target_to_scitypes(df::DataFrame)
    # Coerce the first 13 columns (features) to Continuous
    X = coerce(select(df, 1:13), Count => Continuous)

    # Coerce the 14th column (target) to OrderedFactor
    y = coerce(df[:, 14], OrderedFactor)

    return X, y
end


"""
    map_thal_column!(df::DataFrame) -> DataFrame

This function modifies the `thal` column in the provided DataFrame `df`. It maps the values in the `thal` column as follows:
- Missing values remain unchanged.
- A value of `3` is mapped to `1`.
- A value of `6` is mapped to `2`.
- A value of `7` is mapped to `3`.

# Arguments
- `df::DataFrame`: The DataFrame that contains a column named `thal`.

# Returns
- The modified DataFrame with the `thal` column updated in place.

# Example
```julia
df = DataFrame(thal = [3, 6, 7, missing, 3])
map_thal_column!(df)
# Result: DataFrame with `thal` column values [1, 2, 3, missing, 1]
"""

function map_thal_column!(df::DataFrame)
    df.thal = map(x -> ismissing(x) ? x : x == 3 ? 1 : x == 6 ? 2 : 3, df.thal)
end

function map_heartd_column_1_2_3_4!(df::DataFrame)
    df.heart_disease = map(x -> x == 4 ? 1 : x == 3 ? 1 : x == 2 ? 1 : x, df.heart_disease)
end
