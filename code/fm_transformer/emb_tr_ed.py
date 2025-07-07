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
    name="Flow Matching - Draft 3[Encoder-Decoder with Transformer]"#f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

class TransformerEncoderDecoderEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=5)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, tgt=None, teacher_forcing=True):
        emb = self.embedding(x)
        memory = self.encoder(emb)
        if teacher_forcing and tgt is not None:
            tgt_emb = self.embedding(tgt)
            out = self.decoder(tgt=tgt_emb, memory=memory)
        else:
            bs, seq_len = x.size()
            out_tokens = torch.zeros_like(x)
            out_emb = torch.zeros(bs, seq_len, emb.size(-1), device=emb.device)
            for t in range(seq_len):
                dec_input = out_emb[:, :t+1, :]
                dec_output = self.decoder(tgt=dec_input, memory=memory)
                out_emb[:, t, :] = dec_output[:, t, :]
            out = out_emb
        return self.output_proj(out)

EMB_MODEL_PATH = "scratch/embedding_model_tr.pt"
embedding_model = TransformerEncoderDecoderEmbedding(
    vocab_size=6,
    embedding_dim=embedding_dim,
    num_heads=4,
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
            output = embedding_model(batch, tgt=batch, teacher_forcing=True)
            loss = F.mse_loss(output, embedding_model.embedding(batch))
            loss.backward()
            optimizer.step()
            wandb.log({"Tr_ED/loss": loss.item()})
            if loss.item() < early_stop_threshold:
                stop_training = True
                break
        print(f"Epoch {epoch+1} completed with loss {loss.item():.4f}")
    torch.save(embedding_model.state_dict(), EMB_MODEL_PATH)
    wandb.save(EMB_MODEL_PATH)
    print("Saved trained embedding model.")

embedding_model.eval()
