import torch

from tokenizer.char_tokenizer import CharacterTokenizer
from model.gpt import MiniGPT


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


checkpoint = torch.load(
    "minigpt.pt",
    map_location=DEVICE
)


tokenizer = CharacterTokenizer(
    "".join(checkpoint["chars"])
)

model = MiniGPT(
    vocab_size=checkpoint["vocab_size"],
    context_length=checkpoint["context_length"],
    embed_dim=checkpoint["embed_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()


def generate(
    prompt,
    max_new_tokens=200,
    temperature=1.0
):

    tokens = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(max_new_tokens):

        input_context = input_ids[
            :, -model.context_length:
        ]

        with torch.no_grad():

            logits, _ = model(
                input_context
            )

        logits = logits[:, -1, :]

        logits = logits / temperature

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

    return tokenizer.decode(
        input_ids[0].tolist()
    )


print(
    generate(
        "Artificial",
        max_new_tokens=200,
        temperature=0.8
    )
)