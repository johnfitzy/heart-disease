"""
    clean_value(x, desired_type)

Converts a value `x` to the `desired_type` if possible, or returns `missing` if the conversion fails.

### Arguments
- `x`: The input value to be cleaned. Can be a string, number, or `missing`.
- `desired_type`: The type you want to convert `x` to, e.g., `Int` or `Float64`.

### Behavior
- If `x` is a string, the function strips any leading or trailing whitespace and attempts to parse it as the `desired_type`. If parsing fails, it returns `missing`.
- If `x` is already a number or `missing`, it is returned unchanged.

### Example
```julia
using DataFrames

# Example DataFrame
df = DataFrame(
    a = ["1", "2", "?", "4", "5"],
    b = ["100", "200", "?", "400", "500"],
    c = ["1.5", "2.3", "?", "4.7", "5.1"]
)

# Clean values in the DataFrame columns
df.a = map(x -> clean_value(x, Int), df.a)
df.b = map(x -> clean_value(x, Int), df.b)
df.c = map(x -> clean_value(x, Float64), df.c)

# Resulting DataFrame:
# 5×3 DataFrame
#  Row │ a        b        c
#      │ Int64?   Int64?   Float64?
# ─────┼────────────────────────────
#    1 │ 1        100      1.5
#    2 │ 2        200      2.3
#    3 │ missing  missing  missing
#    4 │ 4        400      4.7
#    5 │ 5        500      5.1
"""

function clean_value(x, desired_type)
    	if x isa AbstractString
			
			# Strip any spaces
        	stripped_x = strip(x)
        	try
            	return parse(desired_type, stripped_x)
        	catch e
            	return missing
        	end
    	else
			# Keep missing or already parsed values as they are
        	return x
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

