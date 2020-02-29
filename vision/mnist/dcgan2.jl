using Base.Iterators: partition
using Flux
using Flux.Data.MNIST
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy, glorot_normal
using Images
using Statistics
using Printf

const BATCH_SIZE = 128
const NOISE_DIM = 100
const EPOCHS = 15
const VERBOSE_FREQ = 500
const ANIMATION_X = 6
const ANIMATION_Y = 6

result_dir = "mnist-dcgan-results"

mutable struct DCGAN
    generator::Chain
    discriminator::Chain
    data::Vector{<: AbstractArray{Float32, 4}}
    animation_noise::AbstractMatrix{Float32}

    train_steps::Int64
end

function DCGAN(; image_vector::Vector{<: AbstractMatrix},
    generator::Chain, discriminator::Chain)

    data = [reshape(reduce(hcat, channelview.(xs)), 28, 28, 1, :) for xs in partition(image_vector, BATCH_SIZE)]
    data = [2f0 .* gpu(Float32.(xs)) .- 1f0 for xs in data]

    animation_noise = randn(Float32, NOISE_DIM, ANIMATION_X * ANIMATION_Y) |> gpu

    DCGAN(generator, discriminator, data, animation_noise, 0)
end

function generator_loss(fake_output)
    loss = mean(logitbinarycrossentropy.(fake_output, 1f0))
    return loss 
end

function discriminator_loss(real_output, fake_output)
    real_loss = mean(logitbinarycrossentropy.(real_output, 1f0))
    fake_loss = mean(logitbinarycrossentropy.(fake_output, 0f0))
    loss = 0.5f0 * (real_loss +  fake_loss)
    return loss
end

function convert_to_image(image_array::Matrix{Float32})
    image_array = @. (image_array + 1f0) / 2f0
    return Gray.(image_array)
end

function save_fake_image(dcgan::DCGAN)
    @eval Flux.istraining() = false
    fake_images = dcgan.generator(dcgan.animation_noise)
    @eval Flux.istraining() = true
    h, w, _, _ = size(fake_images)
    tile_image = Matrix{Float32}(undef, h * ANIMATION_X, w * ANIMATION_Y)
    for n in 0:ANIMATION_X * ANIMATION_Y - 1
        j = n ÷ ANIMATION_X
        i = n % ANIMATION_Y
        tile_image[j * h + 1:(j + 1) * h, i * w + 1:(i + 1) * w] = fake_images[:, :, :, n + 1] |> cpu
    end
    image = convert_to_image(tile_image)
    save(@sprintf("%s/steps_%06d.png", result_dir, dcgan.train_steps), image)
end

function train_discriminator!(dcgan::DCGAN, batch, opt_dscr)
    noise = randn(Float32, NOISE_DIM, BATCH_SIZE) |> gpu
    fake_input = dcgan.generator(noise)
    ps = Flux.params(dcgan.discriminator)
    # Taking gradient
    loss, back = Flux.pullback(ps) do
        discriminator_loss(dcgan.discriminator(batch), dcgan.discriminator(fake_input))
    end
    grad = back(1f0)
    update!(opt_dscr, ps, grad)
    return loss
end

function train_generator!(dcgan::DCGAN, batch, opt_gen)
    noise = randn(Float32, NOISE_DIM, BATCH_SIZE) |> gpu
    ps = Flux.params(dcgan.generator)
    # Taking gradient
    loss, back = Flux.pullback(ps) do
        generator_loss(dcgan.discriminator(dcgan.generator(noise)))
    end
    grad = back(1f0)
    update!(opt_gen, ps, grad)
    return loss
end

function train!(dcgan::DCGAN, opt_gen, opt_dscr)
    for ep in 1:EPOCHS
        @info "Epoch $ep"
        for batch in dcgan.data
            disc_loss = train_discriminator!(dcgan, batch, opt_dscr)
            gen_loss = train_generator!(dcgan, batch, opt_gen)

            if dcgan.train_steps % VERBOSE_FREQ == 0
                @info("Train step $(dcgan.train_steps), Discriminator loss: $(disc_loss), Generator loss: $(gen_loss)")
                # create fake images for animation
                save_fake_image(dcgan)
            end
            dcgan.train_steps += 1
        end
    end
end


function main()
    if !isdir(result_dir)
        mkdir(result_dir)
    end

    generator = Chain(
        Dense(NOISE_DIM, 7 * 7 * 256),
        BatchNorm(7 * 7 * 256, relu),
        x->reshape(x, 7, 7, 256, :),
        ConvTranspose((5, 5), 256 => 128; stride = 1, pad = 2),
        BatchNorm(128, relu),
        ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1),
        BatchNorm(64, relu),
        ConvTranspose((4, 4), 64 => 1, tanh; stride = 2, pad = 1),
        ) |> gpu

    discriminator =  Chain(
        Conv((4, 4), 1 => 64, leakyrelu; stride = 2, pad = 1),
        Dropout(0.25),
        Conv((4, 4), 64 => 128, leakyrelu; stride = 2, pad = 1),
        Dropout(0.25), 
        x->reshape(x, 7 * 7 * 128, :),
        Dense(7 * 7 * 128, 1)) |> gpu

    opt_gen = ADAM(0.0002)
    opt_dscr = ADAM(0.0002)

    dcgan = DCGAN(; image_vector = MNIST.images(),
        generator = generator, discriminator = discriminator)
    train!(dcgan, opt_gen, opt_dscr)

end

main()