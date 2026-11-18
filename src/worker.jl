using SHA, TOML
include("predict.jl")

function load_map(path, n, stride)
    p = Matrix{Float64}(undef, n, n)
    lines = 0
    open(path) do file
        for (row, line) in enumerate(eachline(file))
            lines = row
            mod(row-1, stride) == 0 || continue
            columns = split(line)
            length(columns) == n*stride || error("Unexpected map width")
            p[div(row-1, stride)+1, :] .= parse.(Float64, columns[1:stride:end])
        end
    end
    lines == n*stride || error("Unexpected map height")
    all(x -> 0 <= x <= 1, p) && p == transpose(p) || error("Invalid probability map")
    return p
end

# POSIX clock() gives process CPU time in millionths of a second on Linux/macOS.
process_cpu() = ccall(:clock, Clong, ()) / 1e6

function check_grid(out, p)
    ids = unique(round.(Int, range(1, size(p,1), length=16)))
    reference = loop_extr(p[ids, ids])
    for a in eachindex(ids), b in eachindex(ids), c in eachindex(ids)
        # Include only triples eligible at both resolutions.
        if minimum((d_periodic(a,b,16), d_periodic(b,c,16), d_periodic(a,c,16))) > 1
            out[ids[a], ids[b], ids[c]] == reference[a,b,c] || error("Grid mismatch")
        end
    end
end

function trial(predict, p, bait, plane_path, full_hash)
    GC.gc()
    cpu_start = process_cpu()
    start = time_ns()
    out = predict(p)
    seconds = (time_ns() - start) * 1e-9
    cpu_seconds = process_cpu() - cpu_start
    rss = Sys.maxrss()
    plane = view(out, :, :, bait)
    if !isempty(plane_path)
        open(io -> write(io, plane), plane_path, "w")
    end
    # Hash a zero-copy byte view. At high resolution compare a sparse grid to
    # the serial oracle without allocating or reading a second full tensor.
    digest = full_hash ? bytes2hex(sha256(reinterpret(UInt8, vec(out)))) : ""
    full_hash || check_grid(out, p)
    return (; seconds, cpu_seconds, peak_rss_bytes=Int(rss), digest)
end

function main(args)
    mode, path, ns, step, bait_s, repeats_s, dest = args
    n, stride, bait, repeats = parse.(Int, (ns, step, bait_s, repeats_s))
    p = load_map(path, n, stride)
    if mode == "preview"
        plane = predict_plane!(zeros(n,n), p, bait)
        open(io -> write(io, plane), joinpath(dest, "bait.f64"), "w")
        return
    end
    predict = mode == "serial" ? loop_extr : threaded
    # Compile the same method signatures on a tiny real map before timing.
    tiny = p[1:8, 1:8]
    threaded(tiny) == loop_extr(tiny) || error("Warm-up disagrees with serial")
    rows = []
    for rep in 1:repeats
        plane_path = rep == 1 && mode != "serial" && !isfile(joinpath(dest, "bait.f64")) ?
            joinpath(dest, "bait.f64") : ""
        sample = trial(predict, p, bait, plane_path, n <= 404)
        push!(rows, Dict(string(k) => v for (k,v) in pairs(sample)))
    end
    metrics = Dict("trials" => rows, "julia" => string(VERSION), "cpu" => Sys.cpu_info()[1].model,
                   "threads" => Threads.nthreads(), "mode" => mode)
    TOML.print(stdout, metrics)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
