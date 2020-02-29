using Base.Iterators: partition
using Flux
using Flux: logitbinarycrossentropy, pullback
using Flux.Data.MNIST
using Flux.Optimise: update!
using Images
using Statistics

const BATCH_SIZE = 128
const NOISE_DIM = 100
const EPOCHS = 15
const VERBOSE_FREQ = 100

const SAMPLE_X = 4
const SAMPLE_Y = 6

# Taking vector of images and return minibatch
function make_minibatch(xs)
    ys = reshape(Float32.(reduce(hcat, xs)), 28, 28, 1, :)
    return @. 2f0 * ys - 1f0
end

function train_generator!(gen, dscr, batch, opt_gen)
    noise = randn(Float32, NOISE_DIM, BATCH_SIZE) |> gpu
    ps = params(gen)
    # Taking gradient
    loss, back = pullback(ps) do
        fake_output = dscr(gen(noise))
        mean(logitbinarycrossentropy.(fake_output, 1f0))
    end
    grad = back(1f0)
    update!(opt_gen, ps, grad)
    return loss
end

function train_discriminator!(gen, dscr, batch, opt_dscr)
    noise = randn(Float32, NOISE_DIM, BATCH_SIZE) |> gpu
    fake_input = gen(noise)
    ps = params(gen)
    # Taking gradient
    loss, back = pullback(ps) do
        real_output = dscr(batch)
        fake_output = dscr(fake_input)
        real_loss = mean(logitbinarycrossentropy.(real_output, 1f0))
        fake_loss = mean(logitbinarycrossentropy.(fake_output, 0f0))
        real_loss + fake_loss
    end
    grad = back(1f0)
    update!(opt_dscr, ps, grad)
    return loss
end

function sample(gen)
    noise = [randn(NOISE_DIM, 1) |> gpu for _=1:SAMPLE_X*SAMPLE_Y]
    @eval Flux.istraining() = false
    fake_images = @. cpu(gen(noise))
    @eval Flux.istraining() = true
    reduce(vcat, reduce.(hcat, partition(fake_images, SAMPLE_Y)))
end

function train()
    # Load MNIST images
    images = MNIST.images()
    data = [make_minibatch(xs) |> gpu for xs in partition(images, BATCH_SIZE)]

    # Generator
    generator = Chain(
        Dense(NOISE_DIM, 7 * 7 * 256),
        BatchNorm(7 * 7 * 256, relu),
        x->reshape(x, 7, 7, 256, :),
        ConvTranspose((5, 5), 256 => 128, stride = 1, pad = 2),
        BatchNorm(128, relu),
        ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1),
        BatchNorm(64, relu),
        ConvTranspose((4, 4), 64 => 1, tanh; stride = 2, pad = 1)) |> gpu
    
    # Discriminator
    discriminator = Chain(
        Conv((4, 4), 1 => 64, stride = 2, pad = 1),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.3),
        Conv((4, 4), 64 => 128, stride = 2, pad = 1),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.3),
        x->reshape(x, 7 * 7 * 128, :),
        Dense(7 * 7 * 128, 1)) |> gpu
    
    # Optimizers
    opt_gen = ADAM(0.0002f0)
    opt_dscr = ADAM(0.0002f0)

    for ep in 1:EPOCHS
        @info "Epoch $(ep)"

        train_step = 0
        loss_dscr = 0f0
        loss_gen = 0f0
        for batch in data
            loss_dscr = train_discriminator!(generator,discriminator, batch, opt_dscr)
            loss_gen = train_generator!(generator, discriminator, batch, opt_gen)
        end

        if train_step % VERBOSE_FREQ == 0
            @info "Step $(train_step), Discriminator loss = $(loss_dscr), Generator loss = $(loss_gen)"
        end

        train_step += 1
    end

    # Save generated images
    save("sample_dcgan.png", sample())
end

train()