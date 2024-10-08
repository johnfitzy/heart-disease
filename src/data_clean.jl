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

function clean_value(x, desired_type)
    if x isa AbstractString
        stripped_x = strip(x)
        try
            return parse(desired_type, stripped_x)
        catch e
            return missing
        end
    elseif x isa Number
        return convert(desired_type, x)  # Convert to the desired type if it's a number
    else
        return x  # Return as is for missing or already cleaned values
    end
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

