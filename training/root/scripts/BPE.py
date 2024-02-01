from tokenizers import Tokenizer

#   Wrapper around pre-trained BPE Tokenizer from Cruz et al. at: https://huggingface.co/jcblaise/roberta-tagalog-large/tree/main
class BPETokenizer():
    def __init__(self) -> None:
        self.tokenizer = Tokenizer.from_file("root/tokenizers/tokenizer.json")
    def tokenize(self, doc):
        return self.tokenizer.encode(doc).tokens