# using Revise
# cd(@__DIR__)
using Test

using Base.Threads; println("--> There are ", Threads.nthreads(), " threads")

# Extract options
const DRY_RUN = "-n" in ARGS || "--dryrun" in ARGS
const RUN_ALL = "-a" in ARGS || "--all" in ARGS

# Filter out flags
const RAW_ARGS = filter(arg -> arg ∉ ["-n", "--dryrun", "-a", "--all"], ARGS)

# If RUN_ALL is set, ignore other arguments (set to empty so all tests run)
# Otherwise use the filtered arguments
const TEST_ARGS = RUN_ALL ? String[] : RAW_ARGS

if DRY_RUN
    println("--> Dry run mode active. Tests will be listed but not executed.")
end

if RUN_ALL && !isempty(RAW_ARGS)
    println("--> Running all tests (-a/--all specified), ignoring filters: ", join(RAW_ARGS, " "))
end

# Helper to convert glob pattern to regex
function glob_to_regex(pattern::AbstractString)
    # Escape special regex characters except * and ?
    regex_str = replace(pattern,
        "." => "\\.",
        "+" => "\\+",
        "(" => "\\(",
        ")" => "\\)",
        "[" => "\\[",
        "]" => "\\]",
        "{" => "\\{",
        "}" => "\\}",
        "^" => "\\^",
        "\$" => "\\\$"
    )
    # Convert glob wildcards to regex wildcards
    regex_str = replace(regex_str, "*" => ".*", "?" => ".")
    # Anchor to full string
    return Regex("^" * regex_str * "\$")
end

# Precompute regexes for filtering
const TEST_REGEXES = map(glob_to_regex, TEST_ARGS)

@testset "BifurcationKit" begin
    # Iterate over all items in the test directory
    # Sort to ensure deterministic order
    for item in sort(readdir(@__DIR__))
        dir_path = joinpath(@__DIR__, item)
        
        # Skip runtests.jl explicitely (though isdir usually handles it)
        if item == "runtests.jl"
            continue
        end
        
        # process only directories
        if isdir(dir_path)
            files_to_run = []
            
            # Find all .jl files in this subdirectory
            # Sort files for deterministic order
            for file in sort(readdir(dir_path))
                if endswith(file, ".jl")
                    
                    # Logic to decide if we run this file
                    should_run = isempty(TEST_REGEXES)
                    if !should_run
                        file_path = joinpath(item, file)
                        path_no_ext = joinpath(item, replace(file, ".jl" => ""))
                        
                        for regex in TEST_REGEXES
                            # Check directory name match (e.g. "newton")
                            if !isnothing(match(regex, item))
                                should_run = true
                                break
                            end
                            # Check full file path match (e.g. "newton/test_newton.jl")
                            if !isnothing(match(regex, file_path))
                                should_run = true
                                break
                            end
                            # Check file path without extension (e.g. "newton/test_newton")
                            if !isnothing(match(regex, path_no_ext))
                                should_run = true
                                break
                            end
                        end
                    end

                    if should_run
                        push!(files_to_run, file)
                    end
                end
            end
            
            # If we found relevant files, run them in a testset
            if !isempty(files_to_run)
                @testset "$item" begin
                    for file in files_to_run
                        if DRY_RUN
                            @info "Dry run: would run $item/$file"
                        else
                            @info "Running $item/$file"
                            include(joinpath(dir_path, file))
                        end
                    end
                end
            end
        end
    end
end
