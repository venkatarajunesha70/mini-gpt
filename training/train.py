import torch
from torch.utils.data import DataLoader

from tokenizer.char_tokenizer import CharacterTokenizer
from data.dataset import LanguageModelDataset
from model.gpt import MiniGPT


# -----------------------
# Configuration
# -----------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CONTEXT_LENGTH = 128
BATCH_SIZE = 32

EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4

LEARNING_RATE = 3e-4
EPOCHS = 10


# -----------------------
# Load data
# -----------------------

with open(
    "data/input.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


# -----------------------
# Tokenizer
# -----------------------

tokenizer = CharacterTokenizer(text)

print(
    "Vocabulary size:",
    tokenizer.vocab_size
)


# -----------------------
# Dataset
# -----------------------

dataset = LanguageModelDataset(
    text,
    tokenizer,
    CONTEXT_LENGTH
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# -----------------------
# Model
# -----------------------

model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    context_length=CONTEXT_LENGTH,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS
)

model = model.to(DEVICE)


# -----------------------
# Optimizer
# -----------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------
# Training
# -----------------------

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for step, (x, y) in enumerate(loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits, loss = model(
            x,
            y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:

            print(
                f"Epoch {epoch + 1} "
                f"Step {step} "
                f"Loss {loss.item():.4f}"
            )

    average_loss = (
        total_loss / len(loader)
    )

    print(
        f"Epoch {epoch + 1} "
        f"Average Loss: {average_loss:.4f}"
    )


# -----------------------
# Save model
# -----------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "context_length": CONTEXT_LENGTH,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "chars": tokenizer.chars,
    },
    "minigpt.pt"
)

print("Model saved.")