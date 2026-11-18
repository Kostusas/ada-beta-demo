include("upstream/loop_extr.jl")

# One task owns each output plane. Preserve the arithmetic and circular-neighbour
# exclusion in the authors' loop_extr, including floating-point evaluation order.
function predict_plane!(plane, contacts, i)
    n = size(contacts, 1)
    for j in 1:n, k in 1:n
        if d_periodic(i,j,n) > 1 && d_periodic(j,k,n) > 1 && d_periodic(i,k,n) > 1
            p = contacts[i,j]*contacts[j,k] + contacts[i,k]*contacts[k,j] +
                contacts[j,i]*contacts[i,k] - 2*contacts[i,j]*contacts[j,k]*contacts[k,i]
            plane[k,j] = min(1, p)
        else
            plane[k,j] = 0.0
        end
    end
    return plane
end

function threaded!(out, contacts)
    n = size(contacts, 1)
    Threads.@threads for i in 1:n
        predict_plane!(view(out, :, :, i), contacts, i)
    end
    return out
end

function threaded(contacts)
    n = size(contacts, 1)
    return threaded!(Array{Float64}(undef, n, n, n), contacts)
end
