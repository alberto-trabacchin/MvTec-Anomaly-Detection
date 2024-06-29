from torch import nn

class ConvAutoencoder(nn.Module):
    def __init__(self, embedding_dim):
        super(ConvAutoencoder, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # input: (batch_size, 3, 64, 64)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),     # output: (batch_size, 32, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)      # output: (batch_size, 64, 16, 16)
        )
        
        # Latent space
        self.fc1 = nn.Linear(64 * 16 * 16, embedding_dim)          # input: (batch_size, 64*16*16)
        self.fc2 = nn.Linear(embedding_dim, 64 * 16 * 16)          # output: (batch_size, 64*16*16)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),                                         # output: (batch_size, 32, 32, 32)
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()                                           # output: (batch_size, 3, 64, 64)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 64, 16, 16)
        x = self.decoder(x)
        return x
