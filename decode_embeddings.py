import sys
import pickle
import torch
from cheap.pretrained import CHEAP_shorten_2_dim_128
from cheap.proteins import LatentToSequence


def decode_to_sequence(embeddings: dict[str, list]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = CHEAP_shorten_2_dim_128(return_pipeline=True)
    latent_to_sequence = LatentToSequence().to(device)

    x = torch.stack(embeddings["x"]).to(device)
    masks = torch.stack(embeddings["mask"]).to(device)

    uncompressed = pipeline.decode(x, masks)
    seqs = latent_to_sequence.to_sequence(uncompressed)[-1] # Last element is a list of sequences
    return seqs


if __name__ == "__main__":
    input_file = sys.argv[1]
    with open(input_file, "rb") as f:
        payload = pickle.load(f)

    args = payload["args"]
    output_file = payload["output_file"]

    seqs = decode_to_sequence(args)

    with open(output_file, "wb") as f:
        pickle.dump(seqs, f)
