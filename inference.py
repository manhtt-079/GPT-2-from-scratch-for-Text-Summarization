import argparse
import os
import torch
import torch.nn as nn
from transformers import top_k_top_p_filtering
from trainer import GPT2_TextSum

from config.config import Config

def summary(text: str,
            device: torch.device,
            model: GPT2_TextSum,
            max_seq_len: int = 512,
            summary_max_len: int = 128) -> str:

    tokens = model.tokenizer.encode(text=text, max_length=max_seq_len, truncation=True)[:-1] + [model.tokenizer.sep_token_id]
    tokens = torch.tensor(tokens).to(device).unsqueeze(0)

    sep_idx = tokens.shape[-1] - 1
    with torch.no_grad():
        punc_idx = 8
        punc_count = 0
        for _ in range(summary_max_len):
            last_logit = model(tokens).logits[:, -1]

            filter = top_k_top_p_filtering(last_logit, top_k=50, top_p=1.0)
            props = nn.functional.softmax(filter, dim=-1)
            final_token = torch.multinomial(props, num_samples=1)

            tokens = torch.cat([tokens, final_token], dim=-1)
            token_id = final_token[0, 0].cpu().numpy()

            if token_id == punc_idx:
                punc_count += 1
            if token_id == model.tokenizer.eos_token_id or punc_count >= 3:
                return model.tokenizer.decode(tokens.tolist()[0][sep_idx:])

        return model.tokenizer.decode(tokens.tolist()[0][sep_idx:])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, default='./checkpoint')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--summary_max_len', type=int, default=128)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    args = parser.parse_args()
    
    model = GPT2_TextSum.load_from_checkpoint(args.checkpoint)
    
    while True:
        text = input("Enter text here: ")
        s = summary(text=text,
                    device=device,
                    model=model,
                    max_seq_len=args.max_seq_len,
                    summary_max_len=args.summary_max_len)
        print(s)


if __name__ == '__main__':
    main()
