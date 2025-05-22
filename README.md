# Transformers Implementation from Scratch

This project implements the Transformer architecture from scratch, a powerful model introduced by Vaswani et al. for sequence modeling and transduction tasks like machine translation, text summarization, and more.

The Transformer model is composed of several key components and layers, which are described below along with the mathematical formulas used in each.

## 1. Input Embedding
Each token in the input sequence is mapped to a dense vector representation called an embedding.

```X_{emb} = [ E(x_1), E(x_2), ..., E(x_n) ]```

Where ```E(xᵢ)``` is the learned embedding vector of the i-th token. These vectors are of dimension ```d_model```.

## 2. Positional Encoding
Since the Transformer has no recurrence or convolution, it uses positional encoding to inject information about token positions in the sequence. This uses sine and cosine functions of different frequencies:

```PE(pos, 2i) = sin(pos/10000^(2i/d_model))```

```PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))```

This encoding is added to the input embedding:
```Z₀ = X_embedded + PE```



## 3. Layer Normalization (LayerNorm)
Applied after residual connections to stabilize and accelerate training.

Normalizes the input across the feature dimension:

```μ = (1 / d) * Σ xᵢ```

```σ = sqrt((1 / d) * Σ (xᵢ - μ)² + ε)```

```LayerNorm(x) = γ * (x - μ) / σ + β```

Here, ```γ``` and ```β``` are learnable parameters, and ε is a small value to avoid division by zero.

## 4. Multi-Head Self-Attention
Allows the model to jointly attend to information from different representation subspaces.

### a. Scaled Dot-Product Attention

```Attention(Q, K, V) = softmax((Q × Kᵀ) / sqrt(d_k)) × V```

Where:

```Q```, ```K```, and ```V``` are queries, keys, and values derived from the input

```d_k``` is the dimension of the key vectors (for scaling)

### b. Multi-Head Attention
Multi-head attention allows the model to focus on different parts of the input:

```headᵢ = Attention(Q × Wᵢ^Q, K × Wᵢ^K, V × Wᵢ^V)```

```MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) × W^O```

Where ```Wᵢ^Q```, ```Wᵢ^K```, ```Wᵢ^V```, and ```W^O``` are learned projection matrices.

## 5. Feed-Forward Network (FFN)
Position-wise feed-forward networks are applied independently to each position.

Comprises two linear transformations with a ReLU activation in between:

```FFN(x) = max(0, x × W₁ + b₁) × W₂ + b₂```

## 6. Residual Connections
Residual (skip) connections help in gradient flow and prevent vanishing gradients.

Applied around each sub-layer (multi-head attention and FFN) followed by layer normalization:

```Z' = LayerNorm(x + Sublayer(x))```



