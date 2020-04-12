using Base.Iterators: partition
using CUDAapi
using Distributions
using Flux
using Flux: binarycrossentropy
using Flux.Data: DataLoader
using Images
using Logging: with_logger
using MLDatasets
using Statistics
using TensorBoardLogger: TBLogger, tb_overwrite
using Random

function get_data()
    xtrain, _ = MLDatasets.MNIST.traindata(Float32)
    # MLDatasets uses HWCN format, Flux works with WHCN 
    xtrain = reshape(permutedims(xtrain, (2, 1, 3)), 28^2, :)
    train_loader = DataLoader(xtrain, batchsize=128, shuffle=true)
    train_loader
end

struct Encoder
    linear
    μ
    logσ
    Encoder(input_dim, latent_dim, hidden_dim, device) = new(
        Dense(input_dim, hidden_dim, tanh) |> device,  # linear
        Dense(hidden_dim, latent_dim) |> device,  # μ
        Dense(hidden_dim, latent_dim) |> device,  # logσ
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

function loss(encoder, decoder, x, device)
    μ, logσ = encoder(x)
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ))

    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    logp_x_z = -sum(binarycrossentropy.(decoder(z), x))
    # regularization
    reg = 0.01f0 * sum(x->sum(x.^2), Flux.params(decoder))
    # println((-logp_x_z + kl_q_p) / 128)
    -logp_x_z + kl_q_p + reg
end

function sample(decoder, latent_dim, device)
    y = randn(Float32, latent_dim) |> device
    rand.(Bernoulli.(decoder(y)))
end

img(x) = Gray.(reshape(x, 28, 28))

function train()
    if CUDAapi.has_cuda_gpu()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    loader = get_data()
    
    input_dim = 28^2
    latent_dim = 10
    hidden_dim = 500
    verbose_freq = 500
    
    encoder = Encoder(input_dim, latent_dim, hidden_dim, device)
    decoder = Decoder(input_dim, latent_dim, hidden_dim, device)

    opt = ADAM()
    ps = Flux.params(encoder.linear, encoder.μ, encoder.logσ, decoder)

    # Logging by TensorBoard.jl
    tblogger = TBLogger("logs", tb_overwrite)

    train_steps = 0
    for i = 1:20
        @info "eopch $(i)"
        for x in loader 
            gs = gradient(ps) do
                loss(encoder, decoder, x |> device, device)
            end
            Flux.Optimise.update!(opt, ps, gs)

            # if train_steps % verbose_freq == 0
            train_loss = loss(encoder, decoder, x |> device, device)
            with_logger(tblogger) do
                @info "train" aloss=train_loss
            end
            # end
            train_steps += 1
        end
        s = hcat(img.([sample(decoder, latent_dim, device) for i = 1:10])...)
        save("sample$(i).png", s)
    end      
end

train()