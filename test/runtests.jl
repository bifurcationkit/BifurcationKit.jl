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

function should_run(file_path, args)
    if isempty(args)
        return true
    end

    # file_path is like "folder/file.jl"
    dir, file = splitdir(file_path)
    file_no_ext = replace(file, ".jl" => "")
    path_no_ext = joinpath(dir, file_no_ext)

    for arg in args
        # Convert argument to regex assuming it is a glob pattern
        regex = glob_to_regex(arg)

        # Match directory (e.g. "continuation" matches "continuation")
        # OR "cont*" matches "continuation"
        if !isnothing(match(regex, dir))
            return true
        end

        # Match full path either as "folder/file.jl" or "folder/file"
        # e.g. "cont*/simple*" matches "continuation/simple_continuation.jl"
        if !isnothing(match(regex, file_path)) || !isnothing(match(regex, path_no_ext))
             return true
        end

        # Also support matching just the filename part for convenience?
        # The user requested "plutôt que d'utiliser les regexp... utiliser globbing".
        # Let's stick to full path matching as per previous logic but with globs.
    end
    return false
end


@testset "BifurcationKit" begin
    # Iterate over all items in the test directory
    # Sort to ensure deterministic order
    for item in sort(readdir(@__DIR__))
        dir_path = joinpath(@__DIR__, item)

        # process only directories
        if isdir(dir_path)
            files_to_run = []

            # Find all .jl files in this subdirectory
            # Sort files for deterministic order
            for file in sort(readdir(dir_path))
                if endswith(file, ".jl")
                    file_rel_path = joinpath(item, file)
                    if should_run(file_rel_path, TEST_ARGS)
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
