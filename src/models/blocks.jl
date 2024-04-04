import Flux._big_show
import Flux._show_children
using Flux: _big_finale, _layer_show, _show_layers

cat_on_channel_dim(channel_dim=3) = (h::AbstractArray, x::AbstractArray) -> begin
    cat(x, h, dims=channel_dim)
end

### upsampling and downsampling

function upsample_layer(channels::Pair{<:Integer,<:Integer}; scale=(2, 2), filter=(3,3), stride=(1,1), pad=(1,1))
    Chain(
        Upsample(:nearest; scale),
        Conv(filter, channels; stride, pad)
    )
end

function downsample_layer(channels::Pair{<:Integer,<:Integer}; filter=(4,4), stride=(2,2), pad=(1,1))
    Conv(filter, channels; stride, pad)
end

"""
    ConvEmbed(in => out, emb_channels; groups=8, activation=swish)

A convolutional block that acts on two arguments `(x, emb)`. 
Its output is `activation(norm(conv(x)) .+ embed_layers(emb))` 
where `embed_layers` is shaped so that each value in the embedding channels is mapped to one output channel.
"""
struct ConvEmbed{E,C<:Conv,N,A} <: AbstractParallel
    embed_layers::E
    conv::C
    norm::N
    activation::A
end

Flux.@functor ConvEmbed

function ConvEmbed(channels::Pair{<:Integer,<:Integer}, emb_channels::Int; groups::Int=8, activation=swish, filter=(3,3), stride=(1,1), pad=(1,1))
    out = channels[2]
    embed_layers = Chain(
        swish,
        Dense(emb_channels, out),
    )
    conv = Conv(filter, channels; stride, pad)
    norm = GroupNorm(out, groups)
    ConvEmbed(embed_layers, conv, norm, activation)
end

function (m::ConvEmbed)(x::AbstractArray, emb::AbstractArray)
    h = m.conv(x)
    h = m.norm(h)
    emb_out = m.embed_layers(emb)
    num_ones = length(size(h)) - length(size(emb_out))
    emb_out = reshape(emb_out, (repeat([1], num_ones)..., size(emb_out)...))
    h = h .+ emb_out
    h = m.activation(h)
    h
end

function Base.show(io::IO, m::ConvEmbed)
    print(io, "ConvEmbed(")
    print(io, "embed_layers=", m.embed_layers)
    print(io, ", conv=", m.conv)
    print(io, ", norm=", m.norm)
    print(io, ", activation=", m.activation)
    print(io, ")")
end

function _big_show(io::IO, m::ConvEmbed, indent::Int=0, name=nothing)
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", "ConvEmbed(")
    for layer in [:embed_layers, :conv, :norm, :activation]
        _big_show(io, getproperty(m, layer), indent + 2, layer)
    end
    if indent == 0
        print(io, ") ")
        _big_finale(io, m)
    else
        println(io, " "^indent, ")", ",")
    end
end

"""
    ResBlock(in => out, emb_channels; groups=8, activation=swish)

A residual convolutional block that can optionally change the number of channels.   
Each value in the embedding channels is mapped to one output channel.
"""
struct ResBlock{I<:ConvEmbed,O,S} <: AbstractParallel
    in_layers::I
    out_layers::O
    skip_transform::S
end

Flux.@functor ResBlock

function ResBlock(channels::Pair{<:Integer,<:Integer}, emb_channels::Int; groups::Int=8, activation=swish, filter=(3,3), stride=(1,1), pad=(1,1))
    out_ch = channels[2]
    conv_timestep = ConvEmbed(channels, emb_channels; groups, activation, filter, stride, pad)
    out_layers = Chain(
        Conv(filter, out_ch => out_ch; stride, pad),
        GroupNorm(out_ch, groups),
        activation,
    )
    if channels[1] == channels[2]
        skip_transform = identity
    else
        skip_transform = Conv(filter, channels; stride, pad)
        # skip_transform = Conv(filter, channels; stride, pad)
    end
    ResBlock(conv_timestep, out_layers, skip_transform)
end

function (m::ResBlock)(x::AbstractArray, emb::AbstractArray)
    h = m.in_layers(x, emb)
    h = m.out_layers(h)
    h = h + m.skip_transform(x)
    h
end

function Base.show(io::IO, m::ResBlock)
    print(io, "ResBlock(")
    print(io, "in_layers=", m.in_layers)
    print(io, ", out_layers=", m.out_layers)
    print(io, ", skip_transform=", m.skip_transform)
    print(io, ")")
end

function _big_show(io::IO, m::ResBlock, indent::Int=0, name=nothing)
    println(io, " "^indent, isnothing(name) ? "" : "$name = ", "ResBlock(")
    for layer in [:in_layers, :out_layers, :skip_transform]
        _big_show(io, getproperty(m, layer), indent + 2, layer)
    end
    if indent == 0
        print(io, ") ")
        _big_finale(io, m)
    else
        println(io, " "^indent, ")", ",")
    end
end

### show

for T in [
    :ResBlock, :ConvEmbed
]
    @eval function Base.show(io::IO, m::MIME"text/plain", x::$T)
        if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
            _big_show(io, x)
        elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
            _layer_show(io, x)
        else
            show(io, x)
        end
    end
end
