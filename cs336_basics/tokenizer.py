from typing import Iterable, Iterator, List, Union
import regex as re
from itertools import islice

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.mergeable_ranks: dict[bytes, int] = {b:token_id for token_id, b in vocab.items()}
        
        self.special_tokens_encoder: dict[str, int] = {}
        self.special_tokens_decoder: dict[int, bytes] = {}
        if special_tokens:
            byte_to_id = {v: k for k, v in vocab.items()}
            for token in special_tokens:
                b = token.encode("utf-8")
                if b not in byte_to_id:
                    raise ValueError(f"Special token {token} not found in vocab")
                token_id = byte_to_id[b]
                self.special_tokens_encoder[token] = token_id            
                self.special_tokens_decoder[token_id] = b

        self.allowed_special: set[str] = set(self.special_tokens_encoder.keys())
        
        pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self._pat = re.compile(pat_str)
        self.decoder = vocab.copy()
    
    @staticmethod
    def escaped_to_bytes(s: str) -> bytes:
        """
        反向解析 bytes_to_escaped 的输出：
        解析 \\xNN 序列，还原为原始 bytes。
        """
        out = bytearray()
        i = 0
        while i < len(s):
            if s[i] == "\\" and i + 3 < len(s) and s[i+1] == "x":
                # 形如 \xNN
                hex_str = s[i+2:i+4]
                try:
                    val = int(hex_str, 16)
                    out.append(val)
                    i += 4
                    continue
                except ValueError:
                    # 解析失败，就按普通字符处理
                    pass
            # 普通字符
            out.append(ord(s[i]))
            i += 1
        return bytes(out)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                token_id = int(parts[0])
                token_str = parts[1]
                vocab[token_id] = cls.escaped_to_bytes(token_str)
        
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                a_str, b_str = parts[0], parts[1]
                a_bytes = cls.escaped_to_bytes(a_str)
                b_bytes = cls.escaped_to_bytes(b_str)
                merges.append((a_bytes, b_bytes))
        return cls(vocab, merges, special_tokens)
    
    
    def encode(self, text: str)-> list[int]:
        return self._encode(text, self.allowed_special)
        
    
    def _encode(self, text: str, allowed_special: set)-> List[int]:
        segments = self._split_text_with_special_tokens(text, allowed_special)
        tokens: list[int] = []
        for segment in segments:
            if isinstance(segment, int):
                # This is a special token ID
                tokens.append(segment)
            else:
                for match in self._pat.finditer(segment):
                    piece = match.group()
                    if piece == "":
                        continue
                    tokens.extend(self._encode_bytes(piece.encode('utf-8')))

        return tokens
    
    def _split_text_with_special_tokens(self, text: str, allowed_special: set[str])-> List[Union[str, int]]:
        if not allowed_special:
            return [text]
        sorted_special = sorted(allowed_special, key=len, reverse=True)
        special_pattern = "|".join(re.escape(token) for token in sorted_special)
        regex = re.compile(special_pattern)
        
        segments: list[Union[str, int]] = []
        last_end = 0
        for match in regex.finditer(text):
            start, end = match.span()
            if start > last_end:
                segments.append(text[last_end:start])
            special_token = match.group()
            segments.append(self.special_tokens_encoder[special_token])
            last_end = end
        if last_end < len(text):
            segments.append(text[last_end:])
        return segments
        
    def _encode_bytes(self, byte_string: bytes) -> List[int]:
        """Encode bytes using BPE merging"""
        # Tokenize into individual bytes first
        parts: List[bytes] = [bytes([b]) for b in byte_string]

        while True:
            min_idx = None
            min_rank = None
            # 遍历所有相邻 pair，找最小的 rank（也就是最高优先级的 merge）
            for i, pair in enumerate(zip(parts[:-1], parts[1:])):
                merged_bytes = pair[0] + pair[1]
                rank = self.mergeable_ranks.get(merged_bytes)
                if rank is not None and (min_rank is None or rank < min_rank):
                    min_rank = rank
                    min_idx = i
            if min_rank is None:
                break
            assert min_idx is not None
            # 合并该 pair
            parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
            
        try:
            token_ids = [self.mergeable_ranks[p] for p in parts]
        except KeyError as e:
            raise ValueError(f"Byte sequence {e} not found in vocab/mergeable_ranks") from e
        return token_ids


    def encode_iterable(self, iterable: Iterable[str])->Iterator[int]:
        it = iter(iterable)
        while True:
            batch = list(islice(it, 1000))
            if not batch:
                break
            
            for text in batch:
                if text is None:
                    continue
                
                try:
                    yield from self.encode(text)
                except Exception as e:
                    print(f"Error encoding text: {e}")
                    continue

    
    def decode(self, tokens: List[int])->str:
        bytes_string = self.decode_bytes(tokens)
        try:
            # return bytes_string.decode("utf-8", errors='strict')
            return bytes_string.decode("utf-8", errors='replace')
        except UnicodeDecodeError as e:
            raise ValueError(f"Unable to decode into a valid UTF-8 string: {e}")
    
    def decode_bytes(self, tokens: List[int])->bytes:
        results = bytearray()
        for token in tokens:
            if token in self.decoder:
                results.extend(self.decoder[token])
            elif token in self.special_tokens_decoder:
                results.extend(self.special_tokens_decoder[token])
            else:
                raise ValueError(f"Unknown token: {token}")
        return bytes(results)
    
if __name__ == '__main__':
    model_prefix = "bpe"
    vocab_file = f"./TinyStories_bpe_results/{model_prefix}.vocab"
    merges_filepath = f"./TinyStories_bpe_results/{model_prefix}.merges"
    
    tokenizer = Tokenizer.from_files(vocab_file, merges_filepath, special_tokens=[])
    # print(tokenizer.vocab, tokenizer.merges)

    test_string = "欢迎光临@helloworld🙃"
    encoded_ids = tokenizer.encode(test_string)
    print(f"[MAIN] encoded_ids: {encoded_ids}")
    decoded_string = tokenizer.decode(encoded_ids)
    print(f"[MAIN] decoded_string: {decoded_string}")