from emb import *
import wandb

# Initialize wandb at the start of your script (after your imports)
wandb.login(key="84987f1aa9485c5bac65fe4db16018b4f5a9755e")
wandb.init(
    project="selex-cross-attention",
    config={
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "batch_size": args.batch_size,
        "m": args.m,
        "num_splits": args.num_splits,
        "num_epochs": args.num_epochs,
        "optimizer": "Adam",
        "lr": 1e-3,
    },
    name="Flow Matching - Draft 2[Encoder-Decoder with LSTM]"#f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)


class EncoderDecoderEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=5)

        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True,
            dropout=dropout, bidirectional=True
        )

        self.decoder = nn.LSTM(
            hidden_dim * 2, embedding_dim, num_layers, batch_first=True,
            dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        emb = self.emb_norm(self.embedding(x))  # LayerNorm on embeddings
        enc_out, _ = self.encoder(emb)
        dec_out, _ = self.decoder(enc_out)

        # Residual connection + LayerNorm before projection
        dec_out = self.layer_norm(dec_out + emb)  # Shape match required!

        output = self.output_proj(dec_out)
        return output


EMB_MODEL_PATH = "scratch/embedding_model.pt"
embedding_model = EncoderDecoderEmbedding(
    vocab_size=6,
    embedding_dim=embedding_dim,
    hidden_dim=512,
    num_layers=2
).to(device)

if os.path.exists(EMB_MODEL_PATH):
    embedding_model.load_state_dict(torch.load(EMB_MODEL_PATH))
    print("Loaded trained embedding model.")
else:
    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=1e-3)
    embedding_model.train()
    early_stop_threshold = 1e-3
    stop_training = False
    for epoch in range(10):  # small number of epochs for demonstration
        if stop_training:
            break
        for batch, _ in dataloader:
            optimizer.zero_grad()
            output = embedding_model(batch)
            loss = F.mse_loss(output, embedding_model.embedding(batch))
            loss.backward()
            optimizer.step()
            wandb.log({"ED/loss": loss.item()})
            if loss.item() < early_stop_threshold:
                stop_training = True
                break
        print(f"Epoch {epoch+1} completed with loss {loss.item():.4f}")
    torch.save(embedding_model.state_dict(), EMB_MODEL_PATH)
    wandb.save(EMB_MODEL_PATH)

    print("Saved trained embedding model.")

embedding_model.eval()
