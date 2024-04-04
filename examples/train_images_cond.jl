ENV["CUDA_VISIBLE_DEVICES"] = 1
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

using DenoisingDiffusion
using DenoisingDiffusion: train!, split_validation, batched_loss
include("utilities.jl")

# Load model
model = BSON.load("outputs/Minex_20231231_0647/model_epoch=35.bson")[:model]
diffusion = model |> gpu

### settings
num_timesteps = 500
seed = 2714
dataset = :Minex #:Layers 
data_directory = "../DGMExamples/data/"
output_directory = joinpath("outputs", "$(dataset)_cond_" * Dates.format(now(), "yyyymmdd_HHMM"))
model_channels = 16
learning_rate = 0.0005
batch_size = 128
combine_embeddings = vcat
num_epochs = 35
prob_uncond = 0.0 # 0.2
loss_type = Flux.mse;
to_device = gpu # cpu or gpu

### data
if dataset == :MNIST
    num_classes = 10
    trainset = MNIST(Float32, :train, dir=data_directory);
    norm_data = normalize_neg_one_to_one(reshape(trainset.features, 28, 28, 1, :));
    labels = 2 .+ trainset.targets; # 1->default, 2->0, 3->1, ..., 9->11
    train_x, val_x = split_validation(MersenneTwister(seed), norm_data, labels);

    time_dim = 4 * model_channels
    class_embedding = Flux.Embedding((num_classes + 1) => time_dim)
elseif dataset == :Minex
    trainset = reshape(h5read(joinpath(data_directory, "ore_maps_32.hdf5"), "X")[2:29, 2:29, 1, :], 28,28,1,:)
    # labels = cat(trainset, trainset, dims=3)
    labels = h5read(joinpath(data_directory, "observations.hdf5"), "y")[2:29, 2:29, :, :]
    norm_data = trainset .*2f0 .-1f0 #normalize_neg_one_to_one(reshape(trainset, 28,28,1,:))
    train_x, val_x = split_validation(MersenneTwister(seed), norm_data, labels)

    time_dim = 4 * model_channels
    class_embedding =  Chain(
        Conv((4, 4), 2 => 32, relu; stride = 2, pad = 1),
        Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1),
        Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1),
        Flux.flatten,
        Dense(288, 128, relu),
        Dense(128, time_dim)
    )
    # class_embedding = Flux.Chain(Flux.flatten, Flux.Dense(28*28*2 => 128, relu), Flux.Dense(128=>time_dim))
elseif dataset == :Layers
    trainset = h5read(joinpath("data/", "simple_geology.h5"), "X")[1:28, 1:28, :]
    norm_train = normalize_neg_one_to_one(reshape(trainset, 28,28,1,:))

    labels = h5read(joinpath("data/", "simple_geology.h5"), "Y")[1:28, 1:28, :, :]
    norm_labels = normalize_neg_one_to_one(reshape(labels, 28, 28, 2, :))

    train_x, val_x = split_validation(MersenneTwister(seed), norm_train, norm_labels)

    time_dim = 4 * model_channels
    class_embedding =  Chain(
        Conv((4, 4), 2 => 32, relu; stride = 2, pad = 1),
        Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1),
        Conv((4, 4), 32 => 32, relu; stride = 2, pad = 1),
        Flux.flatten,
        Dense(288, 128, relu),
        Dense(128, time_dim)
    )
else
    throw("$dataset not supported")
end

println("train data:      ", size(train_x[1]), "--", size(train_x[2]))
println("validation data: ", size(val_x[1]), "--", size(val_x[2]))

### model
## create
in_channels = size(train_x[1], 3)
data_shape = size(train_x[1])[1:3]
model = UNetConditioned(in_channels, model_channels, num_timesteps;
    class_embedding=class_embedding,
    block_layer=ResBlock,
    num_blocks_per_level=1,
    block_groups=8,
    channel_multipliers=(1, 2, 4),
    num_attention_heads=4,
    combine_embeddings=combine_embeddings,
)
βs = cosine_beta_schedule(num_timesteps, 0.008)
diffusion = GaussianDiffusion(Vector{Float32}, βs, data_shape, model)
## load
# BSON.@load "outputs\\Pokemon_20230826_1615\\diffusion_opt.bson" diffusion opt_state

display(diffusion.denoise_fn)
println("")

### train
diffusion = diffusion |> to_device

train_data = Flux.DataLoader(train_x |> to_device; batchsize=batch_size, shuffle=true);
val_data = Flux.DataLoader(val_x |> to_device; batchsize=batch_size, shuffle=false);

loss(diffusion, x::AbstractArray, y) = p_losses(diffusion, loss_type, x, y; to_device=to_device)
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
println("done")
# end

println("Calculating initial loss")
val_loss = batched_loss(loss, diffusion, val_data; prob_uncond=prob_uncond)
@printf("\nval loss: %.5f\n", val_loss)

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
    "combine_embeddings" => "$combine_embeddings",
    "parameters" => sum(length, Flux.params(diffusion.denoise_fn)),
    "model_channels" => model_channels,
    "seed" => seed,
    "loss_type" => "$loss_type",
    "learning_rate" => learning_rate,
    "batch_size" => batch_size,
    "prob_uncond" => prob_uncond,
    "optimiser" => "$(typeof(opt).name.wrapper)",
)
open(hyperparameters_path, "w") do f
    JSON.print(f, hyperparameters)
end
println("saved hyperparameters to $hyperparameters_path")

println("Starting training")
start_time = time_ns()
history = train!(
    loss, diffusion, train_data, opt_state, val_data;
    num_epochs=num_epochs, save_after_epoch=true, save_dir=output_directory,
    prob_uncond=prob_uncond
)
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

# all_classes = collect(1:num_classes)
# X0_all = p_sample_loop(diffusion, all_classes; guidance_scale=2.0f0, to_device=to_device);
# X0_all = X0_all |> cpu;
# imgs_all = convert2image(trainset, X0_all[:, :, 1, :]);
# canvas_samples = plot([plot(imgs_all[:, :, i], title="digit=$(i-2)") for i in 1:num_classes]..., ticks=nothing)
# savefig(canvas_samples, joinpath(output_directory, "samples.png"))
# display(canvas_samples)

# for label in 1:num_classes
#     println("press enter for next label")
#     readline()
#     X0 = p_sample_loop(diffusion, 12, label; guidance_scale=2.0f0, to_device=to_device)
#     X0 = X0 |> cpu
#     imgs = convert2image(trainset, X0[:, :, 1, :])
#     canvas = plot([plot(imgs[:, :, i]) for i in 1:12]..., plot_title="label=$label", ticks=nothing)
#     display(canvas)
# end

i=4
# condition = cat(val_x[1][:,:,1:1,1:1], val_x[1][:,:,1:1,1:1], dims=3)
labels = val_x[2][:,:,:,i:i]
p2 = heatmap(labels[:,:,2,1], colorbar=false, clims=(-1,1), axis=false)

all_labels = zeros(28,28,2,16)
all_labels[:,:,:,:] .= labels

all_labels_gpu = all_labels |> gpu

X0 = p_sample_loop(diffusion, 12, all_labels_gpu; guidance_scale=1.0f0, to_device=to_device)

X0 = X0 |> cpu
plots = []
for i=1:12
    # p1 = heatmap(all_labels[:,:,1,i] .* (X0[:, :, 1, i] .+ 1f0) ./ 2f0, colorbar=false, clims=(0,1), axis=false)
    p1 = heatmap( (X0[:, :, 1, i] .+ 1f0) ./ 2f0, colorbar=false, clims=(0,1), axis=false)
    p2 = heatmap(all_labels[:,:,2,i], colorbar=false, clims=(0,1), axis=false)
    p = plot(p1, p2)
    push!(plots, p)
end
plot(plots..., size=(2000,1000))
savefig(joinpath(output_directory, "samples.png"))

println("press enter to finish")
readline()


## Generate some gifs:
Xs, X0s = p_sample_loop_all(diffusion, 16, all_labels_gpu; guidance_scale=1.0f0, to_device=to_device)

Xs = Xs |>cpu
X0s = X0s |>cpu
function combine(imgs::AbstractArray, nrows::Int, ncols::Int, border::Int)
    canvas = zeros(28 * nrows + (nrows+1) * border, 28 * ncols + (ncols+1) * border)
    for i in 1:nrows
        for j in 1:ncols
            left = 28 * (i-1) + 1 + border * i 
            right = 28 * i + border * i
            top = 28 * (j - 1) + 1 + border * j
            bottom = 28 * j + border * j
            canvas[left:right, top:bottom] = imgs[:, :, ncols * (i-1) + j]
        end
    end
    canvas
end

anim_denoise = @animate for i ∈ 1:10:(num_timesteps + 350)
    i = i > num_timesteps ? num_timesteps : i
    imgs = (Xs[:, :, 1, :, i] .+ 1.0 )./ (2.0);
    canvas = combine(imgs, 4, 4, 2)
    p = heatmap(canvas, plot_title="i=$i",colorbar=false, clims=(0,1),axis=false)
end

p2 = heatmap(all_labels[:,:,2,i], colorbar=false, clims=(0,1), axis=false, size=(600,400),)
savefig("observation.png")
gif(anim_denoise, "denoising.gif", fps=15)

