from tokenizers import Tokenizer

#   Wrapper around pretrained BPE Tokenizer from Cruz et al.
class BPETokenizer():
    def __init__(self) -> None:
        self.tokenizer = Tokenizer.from_file("root/tokenizers/tokenizer.json")
    def tokenize(self, doc):
        return self.tokenizer.encode(doc).tokens