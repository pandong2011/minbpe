"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        # 合并次数
        # 新增词表词汇的大小
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8")  # raw bytes
        # 获取encode后的字节序列长度
        ids = list(text_bytes)  # list of integers in range 0..255

        # iteratively merge the most common appear pairs to create new tokens
        # 编码时用
        # 词表本表 字节对 -> 编码后的索引
        # 如(101, 32) -> 256
        # 如(44, 32) -> 257
        # 如(100, 32) -> 258
        merges = {}  # (int, int) -> int
        # 解码时用
        # {索引: 字符}
        # {101:e, 100:d, 50:2 ... }
        # 后续词表
        # {索引: 字符对}
        # {101:e, 100:d, 50:2, 261:20, 263:in, 264:on  ...  }
        vocab = {idx: bytes([idx]) for idx in range(256)}  # int -> bytes
        # 合并新增token
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            # count encode后的字节序列中的 连续pair(相当于two-gram)对
            stats = get_stats(ids)
            # find the pair with the highest count
            # 获取出现频次最高的 pair对
            pair = max(stats, key=stats.get)
            # 新的token id
            # mint a new token: assign it the next available id
            idx = 256 + i
            # 用idx替代 该pair对
            # replace all occurrences of pair in ids with idx
            # 在encode后的字节序列中，用新的idx替代该pair对来缩短ids的长度
            ids = merge(ids, pair, idx)
            # save the merge
            # 将字节对加入词表
            # 编码时用
            merges[pair] = idx
            # 解码时用
            # 🌟vocab[pair[0]] + vocab[pair[1]] 是解码后的字符拼接不是数字相加🌟
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            # occurrences: 出现
            """
            # e空格: 用256表示
            merge 1/256: (101, 32) -> 256; pair (b'e ') had 2981 occurrences
            # 逗号空格: 用257表示
            merge 2/256: (44, 32) -> 257; pair (b', ') had 2961 occurrences
            # d空格: 用258表示
            merge 3/256: (100, 32) -> 258; pair (b'd ') had 2617 occurrences
            ...
            """
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx}; pair ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges  # used in encode()
        self.vocab = vocab  # used in decode()

    def decode(self, ids):
        # given ids (list of integers), return Python string
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8")  # raw bytes
        ids = list(text_bytes)  # list of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break  # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
