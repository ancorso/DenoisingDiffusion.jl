using MLDatasets
using Flux
using CUDA, cuDNN
using Dates
using BSON, JSON
using Printf
using Random
using ProgressMeter
using Plots, Images
using HDF5
using MLUtils

using DenoisingDiffusion
using DenoisingDiffusion: train!, split_validation
include("utilities.jl")
include("load_images.jl")

using Sleipner2019

### settings
num_timesteps = 500
seed = 2714
dataset = :Sleipner # :Sleipner, :Layers  # :MNIST or :Pokemon or Minex or Layers
data_directory = "../DGMExamples/data/"
output_directory = joinpath("outputs", "$(dataset)_" * Dates.format(now(), "yyyymmdd_HHMM"))
model_channels = 16
learning_rate = 0.0005
batch_size = 32
num_epochs = 300
loss_type = Flux.mse;
to_device = gpu # cpu or gpu



### data
if dataset == :MNIST
    trainset = MNIST(Float32, :train, dir=data_directory)
    norm_data = normalize_neg_one_to_one(reshape(trainset.features, 28, 28, 1, :))
    train_x, val_x = split_validation(MersenneTwister(seed), norm_data)
elseif dataset == :Pokemon
    println("loading images")
    data = load_images(data_directory)
    norm_data = normalize_neg_one_to_one(data)
    train_x, val_x = split_validation(MersenneTwister(seed), norm_data)
elseif dataset == :Minex
    trainset = h5read(joinpath(data_directory, "ore_maps_32.hdf5"), "X")[2:29, 2:29, 1, :]
    norm_data = normalize_neg_one_to_one(reshape(trainset, 28, 28, 1, :))
    train_x, val_x = split_validation(MersenneTwister(seed), norm_data)
elseif dataset == :Layers
    trainset = h5read(joinpath("data/", "simple_geology.h5"), "X")[1:28, 1:28, :]
    norm_data = normalize_neg_one_to_one(reshape(trainset, 28, 28, 1, :))
    train_x, val_x = split_validation(MersenneTwister(seed), norm_data)
elseif dataset == :Channels
    trainset = h5read(joinpath("data/", "channels.h5"), "data")[:, :, 1:1, :]
    trainset = cat(trainset, zeros(64, 10, 1, 50000), dims=2)
    norm_data = normalize_neg_one_to_one(trainset)
    train_x, val_x = split_validation(MersenneTwister(seed), norm_data)
elseif dataset == :Mern
    trainset = h5read(joinpath("data/", "mern_x_data.h5"), "x_data")[1:5:200, 1:5:200, 1:1, :]
    norm_data = normalize_neg_one_to_one(trainset)
    train_x, val_x = split_validation(MersenneTwister(seed), norm_data)
elseif dataset == :Sleipner
    filepath = "/scratch/acorso/sleipner_geology/X_32_59_132.h5"
    dataset_dims = size(h5open(filepath, "r")["X"])
    println(dataset_dims)
    Nsamples = dataset_dims[end]
    indices = randperm(MersenneTwister(seed), Nsamples)
    Ntrain = div(Nsamples, 10) * 9
    xt = [(x) -> x[:, :, ceil.(Int, range(1, stop=59, length=32)), :], (x) -> permutedims(x, (2, 3, 4, 1))]
    trainindices, valindices = indices[1:Ntrain], indices[Ntrain+1:end]
    train_x, val_x = GeologyDataset(filepath, trainindices, transforms=xt), GeologyDataset(filepath, valindices, transforms=xt)
    dataset_dims = (size(train_x[1])..., dataset_dims[end])
else
    throw("$dataset not supported")
end


dataset_dims
### model
## create
if dataset !== :Sleipner
    println("train data:      ", size(train_x))
    println("validation data: ", size(val_x))
    in_channels = size(train_x, 3)
    data_shape = size(train_x)[1:3]
    ndims=2
else
    println("datset dims: ", dataset_dims)
    in_channels = dataset_dims[4]
    data_shape = dataset_dims[1:4]
    ndims=3
end

model = UNet(in_channels, model_channels, num_timesteps;
    block_layer=ResBlock,
    num_blocks_per_level=1,
    block_groups=8,
    channel_multipliers=(1, 2, 3),
    num_attention_heads=4,
    ndims = ndims
)
βs = cosine_beta_schedule(num_timesteps, 0.008)
diffusion = GaussianDiffusion(Vector{Float32}, βs, data_shape, model)

## load
# BSON.@load "outputs\\Pokemon_20230826_1615\\diffusion_opt.bson" diffusion opt_state

display(diffusion.denoise_fn)
println("")

### train
diffusion = diffusion |> to_device

if dataset != :Sleipner
    train_data = Flux.DataLoader(train_x |> to_device; batchsize=batch_size, shuffle=true)
    val_data = Flux.DataLoader(val_x |> to_device; batchsize=batch_size, shuffle=false)
else
    train_data = DataLoader(train_x; collate=true, batchsize=batch_size)
    val_data = DataLoader(val_x; collate=true, batchsize=batch_size)
end

loss(diffusion, x::AbstractArray) = p_losses(diffusion, loss_type, x; to_device=to_device)
# if isdefined(Main, :opt_state)
#     opt = extract_rule_from_tree(opt_state)
#     println("existing optimiser: ")
#     println("  ", opt)
#     print("transfering opt_state to device ... ")
#     opt_state = opt_state |> to_device
#     println("done")
# else
println("defining new optimiser")
opt = Adam(learning_rate)
println("  ", opt)
opt_state = Flux.setup(opt, diffusion)
# end

# println("Calculating initial loss")
# val_loss = 0.0
# @showprogress for x in val_data
#     global val_loss
#     val_loss += loss(diffusion, x |> gpu)
# end
# val_loss /= length(val_data)
# @printf("\nval loss: %.5f\n", val_loss)

mkpath(output_directory)
println("made directory: ", output_directory)
hyperparameters_path = joinpath(output_directory, "hyperparameters.json")
output_path = joinpath(output_directory, "diffusion_opt.bson")
history_path = joinpath(output_directory, "history.json")

hyperparameters = Dict(
    "dataset" => "$dataset",
    "num_timesteps" => num_timesteps,
    "data_shape" => "$(diffusion.data_shape)",
    "denoise_fn" => "$(typeof(diffusion.denoise_fn).name.wrapper)",
    "parameters" => sum(length, Flux.params(diffusion.denoise_fn)),
    "model_channels" => model_channels,
    "seed" => seed,
    "loss_type" => "$loss_type",
    "learning_rate" => learning_rate,
    "batch_size" => batch_size,
    "optimiser" => "$(typeof(opt).name.wrapper)",
)
open(hyperparameters_path, "w") do f
    JSON.print(f, hyperparameters)
end
println("saved hyperparameters to $hyperparameters_path")

println("Starting training")
start_time = time_ns()
history = train!(loss, diffusion, train_data, opt_state, val_data; num_epochs=num_epochs, save_after_epoch=true, save_dir=output_directory, Nplots=12, plot_fn=x -> plot_geology(permutedims(x, (4, 1, 2, 3))))
end_time = time_ns() - start_time
println("\ndone training")
@printf "time taken: %.2fs\n" end_time / 1e9

### save results
open(history_path, "w") do f
    JSON.print(f, history)
end
println("saved history to $history_path")

# let diffusion = cpu(diffusion), opt_state = cpu(opt_state)
#     # save opt_state in case want to resume training
#     BSON.bson(
#         output_path, 
#         Dict(
#             :diffusion => diffusion, 
#             :opt_state => opt_state
#         )
#     )
# end
# println("saved model to $output_path")

### plot results

canvas_train = plot(
    1:length(history["mean_batch_loss"]), history["mean_batch_loss"], label="mean batch_loss",
    xlabel="epoch",
    ylabel="loss",
    legend=:right, # :best, :right
    ylims=(0, Inf),
)
plot!(canvas_train, 1:length(history["val_loss"]), history["val_loss"], label="val_loss")
savefig(canvas_train, joinpath(output_directory, "history.png"))
display(canvas_train)


# if dataset == :MNIST || :Minex
#     imgs = convert2image(trainset, X0[:, :, 1, :])
# elseif dataset == :Pokemon
#     for i in 1:12
#         X0[:, :, :, i] = normalize_zero_to_one(X0[:, :, :, i])
#     end
#     imgs = img_WHC_to_rgb(X0)
# end
heatmap(train_x[:, :, 1, 1])
clims = (maximum(train_x), minimum(train_x))
plot([heatmap(train_x[:, :, 1, i], clims=(-1, 1), colorbar=false, axis=false, margin=0Plots.mm) for i in 1:12]...)

X0 = p_sample_loop(diffusion, 12; to_device=to_device)
X0 = cpu(X0)
canvas_samples = plot([heatmap(X0[:, :, 1, i], clims=(-1, 1), colorbar=false, axis=false, margin=0Plots.mm) for i in 1:12]...)
savefig(canvas_samples, joinpath(output_directory, "samples.png"))
display(canvas_samples)

println("press enter to finish")
readline()


model = BSON.load("outputs/Mern_20240301_1800/model_epoch=300.bson")[:model]
diffusion = model |> gpu

## Generate some gifs:
Xs, X0s = p_sample_loop_all(diffusion, 16; to_device=to_device)

Nx = 40
Ny = 40

Xs = Xs |> cpu
X0s = X0s |> cpu
function combine(imgs::AbstractArray, nrows::Int, ncols::Int, border::Int)
    canvas = zeros(Nx * nrows + (nrows + 1) * border, Ny * ncols + (ncols + 1) * border)
    for i in 1:nrows
        for j in 1:ncols
            left = Nx * (i - 1) + 1 + border * i
            right = Nx * i + border * i
            top = Ny * (j - 1) + 1 + border * j
            bottom = Ny * j + border * j
            canvas[left:right, top:bottom] = imgs[:, 1:Ny, ncols*(i-1)+j]
        end
    end
    canvas
end

anim_denoise = @animate for i ∈ 1:10:(num_timesteps+100)
    i = i > num_timesteps ? num_timesteps : i
    imgs = (Xs[:, :, 1, :, i] .+ 1.0) ./ (2.0)
    canvas = combine(imgs, 4, 4, 2)
    p = heatmap(canvas, plot_title="i=$i", colorbar=false, clims=(0, 1), axis=false, cmap=:viridis)
end

p2 = heatmap(all_labels[:, :, 2, i], colorbar=false, clims=(0, 1), axis=false, size=(600, 400),)
savefig("observation.png")
gif(anim_denoise, "stratigraphy_denoising.gif", fps=10)
