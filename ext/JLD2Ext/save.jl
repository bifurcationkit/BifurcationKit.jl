"""
Save solution / data in JLD2 file
- `filename` is for example "example.jld2"
- `sol` is the solution
- `p` is the parameter
- `i` is the index of the solution to be saved
"""
function save_to_file(iter::AbstractContinuationIterable, sol, p, i::Int64, br::ContResult)
    if iter.contparams.save_to_file == false; return nothing; end
    filename = iter.filename
    # this allows to save two branches forward/backward in case
    # bothside = true is passed to continuation
    fd = iter.contparams.ds >=0 ? "fw" : "bw"
    # create a group in the JLD format
    jldopen(filename*".jld2", "a+") do file
        if haskey(file, "sol-$fd-$i")
            delete!(file, "sol-$fd-$i")
        end
        mygroup = JLD2.Group(file, "sol-$fd-$i")
        mygroup["sol"] = sol
        mygroup["param"] = p
    end

    jldopen(filename*"-branch.jld2", "a+") do file
        if haskey(file, "branch"*fd)
            delete!(file, "branch"*fd)
        end
        file["branch"*fd] = br
    end
end

# final save of branch, in case bothside = true is used
function save_to_file(iter::AbstractContinuationIterable, br::ContResult)
    if iter.contparams.save_to_file == false; return nothing; end
    filename = iter.filename
    jldopen(filename*"-branch.jld2", "a+") do file
        if haskey(file, "branchfw")
            delete!(file, "branchfw")
        end
        if haskey(file, "branchbw")
            delete!(file, "branchbw")
        end
        if haskey(file, "branch")
            delete!(file, "branch")
        end
        file["branch"] = br
    end
end