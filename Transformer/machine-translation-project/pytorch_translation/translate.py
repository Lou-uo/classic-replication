import argparse

import torch

from datasets import Tokenizer


def predict(opt):
    device = "cpu"  # 强制使用CPU
    model = torch.load("runs/exp/weights/best.pt", map_location=device, weights_only=False)
    model.device = device  # 确保模型的device属性为CPU
    en_tokenizer = Tokenizer(f"{opt.vocab}/en.vec", is_en=True)
    ch_tokenizer = Tokenizer(f"{opt.vocab}/ch.vec", is_en=False)
    
    import string
    
    while True:
        s = input("请输入英文:")
        # 移除标点符号
        s_clean = s.translate(str.maketrans('', '', string.punctuation))
        translation = model.translate(s_clean, en_tokenizer, ch_tokenizer)
        print(translation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--vocab', default='datas')
    opt = parser.parse_args()
    predict(opt)
