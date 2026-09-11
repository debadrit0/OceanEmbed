import torch
import torch.nn as nn


class OceanEmbedEmbeddingModel(nn.Module):
    """
    OceanEmbed CNN-based satellite/ocean embedding model.

    INPUT
    -----
    [batch, 7, 9, 9]

    Channels
    --------
    0 = SST
    1 = SSS
    2 = SSH / SLA
    3 = U surface current
    4 = V surface current
    5 = U surface wind
    6 = V surface wind

    ARCHITECTURE
    ------------
    Surface patch
        ->
    CNN spatial encoder
        ->
    Compact 64-dimensional latent ocean embedding
        ->
    Temperature reconstruction head
        ->
    15 depth temperatures

    OUTPUT
    ------
    [batch, 15]

    Depths
    ------
    0, 5, 10, 20, 30, 50, 75, 100,
    125, 150, 200, 300, 500, 700, 1000 m
    """

    EMBEDDING_DIM = 64

    OUTPUT_DIM = 15

    def __init__(self):
        super().__init__()

        # ====================================================
        # CNN ENCODER
        # ====================================================

        self.encoder_features = nn.Sequential(

            nn.Conv2d(
                in_channels=7,
                out_channels=24,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=24,
                out_channels=48,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )


        # ====================================================
        # LATENT OCEAN EMBEDDING
        # ====================================================

        self.embedding_layer = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64,
                self.EMBEDDING_DIM
            ),

            nn.ReLU(inplace=True)
        )


        # ====================================================
        # TEMPERATURE RECONSTRUCTION HEAD
        # ====================================================

        self.reconstruction_head = nn.Sequential(

            nn.Linear(
                self.EMBEDDING_DIM,
                48
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                48,
                32
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                32,
                self.OUTPUT_DIM
            )
        )


    # ========================================================
    # ENCODER
    # ========================================================

    def encode(self, x):
        """
        Convert a 7x9x9 surface observation patch
        into a compact latent ocean embedding.

        Returns
        -------
        Tensor of shape [batch, 64]
        """

        features = self.encoder_features(
            x
        )

        embedding = self.embedding_layer(
            features
        )

        return embedding


    # ========================================================
    # RECONSTRUCTION
    # ========================================================

    def reconstruct(self, embedding):
        """
        Reconstruct the 15-depth temperature profile
        from the latent ocean embedding.
        """

        return self.reconstruction_head(
            embedding
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):
        """
        Complete surface -> embedding -> temperature
        reconstruction pipeline.
        """

        embedding = self.encode(
            x
        )

        temperature = self.reconstruct(
            embedding
        )

        return temperature


    # ========================================================
    # EMBEDDING ACCESS
    # ========================================================

    def get_embedding(self, x):
        """
        Convenience method for extracting the learned
        OceanEmbed latent representation.

        Returns
        -------
        Tensor of shape [batch, 64]
        """

        return self.encode(
            x
        )


# ============================================================
# MODEL SELF TEST
# ============================================================

if __name__ == "__main__":

    model = OceanEmbedEmbeddingModel()

    test_input = torch.randn(
        4,
        7,
        9,
        9
    )

    output = model(
        test_input
    )

    embedding = model.get_embedding(
        test_input
    )


    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )


    print()
    print("=" * 70)
    print("OCEANEMBED EMBEDDING MODEL TEST")
    print("=" * 70)

    print(
        "Input shape      :",
        tuple(test_input.shape)
    )

    print(
        "Embedding shape  :",
        tuple(embedding.shape)
    )

    print(
        "Output shape     :",
        tuple(output.shape)
    )

    print(
        "Embedding size   :",
        model.EMBEDDING_DIM
    )

    print(
        "Parameters       :",
        parameter_count
    )

    print("=" * 70)

    print(
        "Model test successful."
    )