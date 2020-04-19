# Variational Autoencoder(VAE)
#
# Auto-Encoding Variational Bayes
# Diederik P Kingma, Max Welling
# https://arxiv.org/abs/1312.6114

using Base.Iterators: partition
using BSON
using CUDAapi: has_cuda_gpu
using DrWatson: savename, struct2dict
using Flux
using Flux: binarycrossentropy
using Flux.Data: DataLoader
using Images
using Logging: with_logger
using MLDatasets
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using TensorBoardLogger: TBLogger, tb_overwrite
using Random

# load MNIST images and return loader
function get_data(batch_size)
    xtrain, _ = MLDatasets.MNIST.traindata(Float32)
    # MLDatasets uses HWCN format, Flux works with WHCN 
    xtrain = reshape(permutedims(xtrain, (2, 1, 3)), 28^2, :)
    train_loader = DataLoader(xtrain, batchsize = batch_size, shuffle=true)
    train_loader
end

struct Encoder
    linear
    μ
    logσ
    Encoder(input_dim, latent_dim, hidden_dim, device) = new(
        Dense(input_dim, hidden_dim, tanh) |> device,   # linear
        Dense(hidden_dim, latent_dim) |> device,        # μ
        Dense(hidden_dim, latent_dim) |> device,        # logσ
    )
end

function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end

Decoder(input_dim, latent_dim, hidden_dim, device) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim, sigmoid)
) |> device

function model_loss(encoder, decoder, λ, x, device)
    μ, logσ = encoder(x)
    len = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len

    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    logp_x_z = -sum(binarycrossentropy.(decoder(z), x)) / len
    # regularization
    reg = λ * sum(x->sum(x.^2), Flux.params(decoder))
    
    -logp_x_z + kl_q_p + reg
end

function generate_image(decoder, latent_dim, sample_size, device)
    x = randn(Float32, sample_size, latent_dim) |> device
    samples = mapslices(decoder, x, dims = 1)
    Gray.(reshape(samples, 28, :))
end

# arguments for the `train` function 
@with_kw mutable struct Args
    η = 1e-3            # learning rate
    λ = 0.01f0          # regularization paramater
    batch_size = 128    # batch size
    sample_size = 10    # sampling size for output    
    epochs = 20         # number of epochs
    seed = 0            # random seed
    cuda = true         # use GPU
    input_dim = 28^2    # image size
    latent_dim = 10     # latent dimension
    hidden_dim = 500    # hidden dimension
    verbose_freq = 10   # logging for every verbose_freq iterations
    tblogger = false    # log training with tensorboard
    save_path = nothing # results path.
end

function train(; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && has_cuda_gpu()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load MNIST images
    loader = get_data(args.batch_size)
    
    # initialize encoder and decoder
    encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim, device)
    decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim, device)

    # ADAM optimizer
    opt = ADAM(args.η)
    
    # parameters
    ps = Flux.params(encoder.linear, encoder.μ, encoder.logσ, decoder)

    # logging by TensorBoard.jl
    tblogger = TBLogger(args.save_path, tb_overwrite)

    # save directory
    if args.save_path == nothing
        experiment_folder = savename("vae", args, scientific=4, accesses=[:batch_size, :η, :λ, :seed])
        args.save_path = joinpath("runs", experiment_folder)
    end

    # training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for x in loader 
            loss, back = Flux.pullback(ps) do
                model_loss(encoder, decoder, args.λ, x |> device, device)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            # progress meter
            next!(progress; showvalues = [(:loss, loss)]) 

            # logging
            if args.tblogger && train_steps % args.verbose_freq == 0
                with_logger(tblogger) do
                    @info "train" loss=loss
                end
            end
            train_steps += 1
        end
        # save image
        s = generate_image(decoder, args.latent_dim, args.sample_size, device)
        image_path = "sample$(epoch).png"
        save(image_path, s)
        @info "Image saved: $(image_path)"
    end

    # save models
    !ispath(args.save_path) && mkpath(args.save_path)
    model_path = joinpath(args.save_path, "model.bson") 
    let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args)
        BSON.@save model_path encoder decoder args
        @info "Model saved: $(model_path)"
    end
end

function plot_result()
    BSON.@load "logs/model.bson" encoder decoder args
    args = Args(; args...)
    device = args.cuda && has_cuda_gpu() ? gpu : cpu
    encoder, decoder = encoder |> device, decoder |> device
end

# train()
