# A session-local worker. Emit tab-separated events and retain only one cube.
include("worker.jl")

function event(values...)
    println(join(values, '\t'))
    flush(stdout)
end

function measure(p, seconds, directory, reference, bait)
    n = size(p, 1)
    tiny = p[1:16, 1:16]
    threaded(tiny) == loop_extr(tiny) || error("Serial warm-up mismatch")
    out = Array{Float64}(undef, n, n, n)
    threaded!(out, p)  # compile, allocate physical pages and warm the chosen resolution
    event("phase", "Predicting triplets")
    passes = 0
    total = 0.0
    times = Float64[]
    last_update = 0.0
    cpu_start = process_cpu()
    wall_start = time_ns()
    while total < seconds
        start = time_ns()
        threaded!(out, p)
        elapsed = (time_ns() - start) / 1e9
        push!(times, elapsed)
        passes += 1
        total += elapsed
        if total - last_update >= 0.4 || total >= seconds
            event("progress", passes, total)
            last_update = total
        end
    end
    busy_cores = (process_cpu() - cpu_start) / ((time_ns()-wall_start)/1e9)
    event("phase", "Checking numerical result")
    # Hash outside the timed window, once per worker. All thread counts at
    # the same resolution must produce the same digest.
    digest = ""
    if n <= 404
        digest = bytes2hex(sha256(reinterpret(UInt8, vec(out))))
    else
        check_grid(out, p)
    end
    if n == 404
        saved = read!(reference, Matrix{Float64}(undef, n, n))
        view(out, :, :, bait) == saved || error("Published bait-200 plane mismatch")
    end
    if n == 101
        out == loop_extr(p) || error("Original serial predictor mismatch")
    end
    sort!(times)
    med = (times[(passes+1)÷2] + times[(passes+2)÷2]) / 2
    plane = joinpath(directory, "bait-$(n).f64")
    if !isfile(plane)
        open(io -> write(io, view(out, :, :, bait)), plane * ".tmp", "w")
        mv(plane * ".tmp", plane)
    end
    event("done", passes, total, med, busy_cores, Sys.maxrss(), digest, Threads.nthreads(), VERSION, Sys.cpu_info()[1].model)
end

function live(args)
    path, ns, step, duration, directory, reference, bs = args
    n, stride, bait = parse.(Int, (ns, step, bs))
    event("phase", "Loading contacts and warming workers")
    measure(load_map(path, n, stride), parse(Float64, duration), directory, reference, bait)
end

live(ARGS)
