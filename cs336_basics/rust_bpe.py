# from rust-bpe by karpathy (https://github.com/karpathy/nanochat)
from typing import Iterator
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import heapq
logger = logging.getLogger(__name__)

Pair = tuple[int, int] # 某一个token对 (100, 200)

class Word:
    """
    Represents a tokenized word as a sequence of token IDs.
    Optimized for incremental BPE merge operations.
    """

    def __init__(self, ids: list[int]):
        self.ids = ids

    def pairs(self) -> Iterator[Pair]:
        """Yield all adjacent (left, right) pairs in the word."""
        for i in range(len(self.ids) - 1):
            yield (self.ids[i], self.ids[i + 1])

    def merge_pair(self, pair: Pair, new_id: int) -> list[tuple[Pair, int]]:
        """
        Merge all non-overlapping occurrences of `pair` -> `new_id`.
        
        Returns a list of local pair-count deltas for THIS word only:
          - (pair, -1): a pair was removed
          - (pair, +1): a new pair was created
        
        This incremental approach avoids re-scanning the entire word
        to recount all pairs, which is critical for performance.
        
        NOTE: deliberately avoids a dict in the hot loop; deltas are
        aggregated later at the corpus level.
        """
        a, b = pair
        n = len(self.ids)
        if n < 2:
            return []

        out: list[int] = []
        # Pre-allocate rough capacity (most words have few merges)
        deltas: list[tuple[Pair, int]] = []

        i = 0
        while i < n:
            # Check if current position matches the target pair
            if i + 1 < n and self.ids[i] == a and self.ids[i + 1] == b:
                # Neighbors before/after the merge region
                left = out[-1] if out else None
                right = self.ids[i + 2] if i + 2 < n else None

                # --- Update pair counts (incremental deltas) ---
                # Left boundary: (left, a) disappears, (left, new_id) appears
                if left is not None:
                    deltas.append(((left, a), -1))
                    deltas.append(((left, new_id), 1))
                # The merged pair itself disappears
                deltas.append(((a, b), -1))

                # Right boundary: (b, right) disappears, (new_id, right) appears
                if right is not None:
                    deltas.append(((b, right), -1))
                    deltas.append(((new_id, right), 1))
                # --- End delta updates ---

                # Emit the merged token
                out.append(new_id)
                i += 2  # Skip both 'a' and 'b'
            else:
                # No match: copy token as-is
                out.append(self.ids[i])
                i += 1
        # Replace the word's token sequence with the merged version
        self.ids = out  # 输出合并结果
        """
        delta = [
            ((left, a), -1),  # 这个pair少了一次
            ((left, new_id), +1),  # 这个pair新增一次
            ((a, b), -1),  # 被合并的pair消失
            ((b, right), -1),
            ((new_id, right), +1)
        ]
        """
        return deltas


@dataclass(order=False) # 禁用自动生成的比较方法，我们手动实现
class MergeJob:
    """
    Represents a candidate merge operation for the BPE priority queue.
    
    Sorting rules (for max-heap behavior with deterministic tie-breaking):
      1. Higher count first (max-heap)
      2. 对于频率相同的pair: 
         >>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
            ('BA', 'A')
    """
    pair: Pair
    count: int
    vocab: dict[int, bytes]
    pos: set[int] = field(default_factory=set)  # word indices where this pair may occur
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MergeJob):
            return NotImplemented
        return self.count == other.count and self.pair == other.pair

    def __hash__(self) -> int:
        # 使用pair和count生成哈希值
        return hash((self.pair, self.count))

    def __lt__(self, other: 'MergeJob') -> bool:
        if self.count != other.count:
            # 希望 count 大的排前面
            return self.count > other.count
        else:
            # Count 相同时
            return (self.vocab[self.pair[0]], self.vocab[self.pair[1]]) > \
                   (other.vocab[other.pair[0]], other.vocab[other.pair[1]])


def count_pairs_parallel(words, counts) -> tuple[dict[Pair, int], dict[Pair, set[int]]]:
    """
    words = [
        Word([97, 98, 99]),   # "abc"
        Word([97, 98, 100]),  # "abd"
        Word([98, 99, 100]),  # "bcd"
    ]
    counts = [10, 5, 3]

    假设：
    索引 0: "abc" (counts=10) → 产生的 pairs: (97,98), (98,99)
    索引 1: "abd" (counts=5) → 产生的 pairs: (97,98), (98,100)
    索引 2: "bcd" (counts=3) → 产生的 pairs: (98,99), (99,100)

    执行后
    pair_counts = {
        (97,98): 15,  # 来自索引0(10) + 索引1(5)
        (98,99): 13,  # 来自索引0(10) + 索引2(3)
        (98,100): 5,  # 来自索引1(5)
        (99,100): 3   # 来自索引2(3)
    }

    where_to_update = {
        (97,98): {0, 1},   # 这对出现在单词0和1中
        (98,99): {0, 2},   # 出现在单词0和2中
        (98,100): {1},     # 只出现在单词1中
        (99,100): {2}      # 只出现在单词2中
    }
    """
    pair_counts = defaultdict(int)  # int指定的是值的默认类型
    where_to_update = defaultdict(set)

    for i, w in enumerate(words):
        if len(w.ids) < 2 or counts[i] == 0:
            continue
        for a, b in w.pairs():
            pair = (a, b)
            pair_counts[pair] += counts[i]
            where_to_update[pair].add(i)
    return pair_counts, where_to_update


class Tokenizer:
    """BPE Tokenizer with incremental training."""
    
    def __init__(self):
        self.merges: dict[Pair, int] = {}  # {pair: new_token_id}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}  # base byte vocab
    
    def train_core_incremental(
        self,
        words: list[Word],
        counts: list[int],
        vocab_size: int,
        not256: int
    ) -> None:
        """
        Core incremental BPE training given unique words and their counts.
        
        Args:
            words: List of Word objects (one per unique token sequence)
            counts: Parallel list of occurrence counts for each word
            vocab_size: Target vocabulary size (must be >= 256)
        """
        assert vocab_size >= 256, "vocab_size must be at least 256"
        # not256 = initial_256 + special_tokens
        num_merges = vocab_size - not256
        logger.info(f"Starting BPE training: {num_merges} merges to compute")
        self.merges.clear()
        
        # ---- Initial pair_counts and where_to_update (parallel) ----
        logger.info(f"Computing initial pair counts from {len(words)} unique sequences")
        # pair_counts: dict[Pair, int]
        # where_to_update: dict[Pair, Set[int]]
        pair_counts, where_to_update = count_pairs_parallel(words, counts)
        
        # ---- Build heap (max-heap via inverted __lt__ in MergeJob) ----
        logger.info(f"Building heap with {len(pair_counts)} unique pairs")
        heap: list[MergeJob] = []
        
        for pair, pos in where_to_update.items():
            count = pair_counts.get(pair, 0)
            if count > 0:
                heapq.heappush(heap, MergeJob(
                    pair=pair,
                    count=count,
                    vocab=self.vocab,
                    pos=pos.copy(),  # copy to avoid aliasing
                ))
        
        # ---- Merge loop ----
        logger.info("Starting merge loop")
        merges_done = 0
        last_log_percent = 0
        
        while merges_done < num_merges:
            # Pop best candidate
            if not heap:
                logger.warning("Heap exhausted before reaching target vocab size")
                break
                
            top = heapq.heappop(heap)

            # ---- Lazy refresh: skip stale entries ----
            current_count = pair_counts.get(top.pair, 0)
            if top.count != current_count:
                # Entry is stale: update count and re-push if still valid
                if current_count > 0:
                    top.count = current_count
                    heapq.heappush(heap, top)
                continue
            
            if top.count == 0:
                # No more positive-count pairs available
                break
            
            # ---- Record the merge ----
            new_id = 256 + merges_done
            self.merges[top.pair] = new_id
            
            # Also update vocab for byte representation (optional, for decoding)
            a, b = top.pair
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            
            # ---- Apply merge to affected words and collect deltas ----
            local_pos_updates: dict[Pair, set[int]] = defaultdict(set)
            
            for word_idx in top.pos:
                word = words[word_idx]
                word_count = counts[word_idx]
                
                # Apply merge and get incremental deltas
                changes: list[tuple[Pair, int]] = word.merge_pair(top.pair, new_id)
                
                # Update global pair counts with weighted deltas
                for pair, delta in changes:
                    delta_total = delta * word_count
                    if delta_total != 0:
                        old_count = pair_counts.get(pair, 0)
                        new_count = old_count + delta_total
                        pair_counts[pair] = new_count
                        
                        # Track which words now contain newly-created pairs
                        if delta > 0:
                            local_pos_updates[pair].add(word_idx)
            
            # ---- Push updated pairs back to heap ----
            for pair, pos in local_pos_updates.items():
                cnt = pair_counts.get(pair, 0)
                if cnt > 0:
                    heapq.heappush(heap, MergeJob(
                        pair=pair,
                        count=cnt,
                        vocab=self.vocab,
                        pos=pos,
                    ))
            
            merges_done += 1
            
            # ---- Progress logging every 1% ----
            current_percent = (merges_done * 100) // num_merges
            if current_percent > last_log_percent:
                logger.info(
                    f"Progress: {current_percent}% ({merges_done}/{num_merges} merges) - "
                    f"Last merge: {top.pair} -> {new_id} (frequency: {top.count})"
                )
                last_log_percent = current_percent
        
        logger.info(f"Finished training: {merges_done} merges completed")
    
    def encode(self, text: str) -> list[int]:
        """
        Encode a string into token IDs using the learned merges.
        
        Args:
            text: Input string to encode
            
        Returns:
            List of token IDs
        """
        # Convert text to bytes
        tokens = list(text.encode('utf-8'))
        
        # Apply merges greedily
        while len(tokens) >= 2:
            # Find the best pair to merge (lowest priority)
            best_pair = None
            best_score = float('inf')
            best_pos = -1
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < best_score:
                        best_score = rank
                        best_pair = pair
                        best_pos = i
            
            if best_pair is None:
                break
            
            # Perform the merge
            new_id = self.merges[best_pair]
            tokens[best_pos] = new_id
            del tokens[best_pos + 1]
        
        return tokens
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back to string.
        
        Args:
            token_ids: List of token IDs to decode
            
        Returns:
            Decoded string
        """
        # Build token bytes (lazy build if needed)
        token_bytes = {i: bytes([i]) for i in range(256)}
        
        # Add merge results
        for (a, b), new_id in self.merges.items():
            if a not in token_bytes:
                token_bytes[a] = self._build_token_bytes(a)
            if b not in token_bytes:
                token_bytes[b] = self._build_token_bytes(b)
            token_bytes[new_id] = token_bytes[a] + token_bytes[b]
        
        # Concatenate and decode
        result_bytes = b''.join(token_bytes.get(tid, b'') for tid in token_ids)
        return result_bytes.decode('utf-8', errors='replace')
    
    def _build_token_bytes(self, token_id: int) -> bytes:
        """Recursively build bytes for a merged token."""
        if token_id < 256:
            return bytes([token_id])
        
        # Find the merge that created this token
        for (a, b), tid in self.merges.items():
            if tid == token_id:
                return self._build_token_bytes(a) + self._build_token_bytes(b)
        
        raise ValueError(f"Token ID {token_id} not found in merges")


# Example usage
def test01():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create tokenizer
    tokenizer = Tokenizer()
    
    # Example words with frequencies
    words = [
        Word([97, 98, 99]),  # "abc"
        Word([97, 98, 100]),  # "abd"
        Word([98, 99, 100]),  # "bcd"
    ]
    counts = [10, 5, 3]
    
    # Train to vocabulary size 260 (4 merges)
    tokenizer.train_core_incremental(words, counts, vocab_size=260)
    
    # Encode some text
    encoded = tokenizer.encode("abc")
    print(f"Encoded: {encoded}")
    
    # Decode back
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")


def test_frequency_tie_breaking():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    def get_lex_order(pairs):
        """返回按字典序降序排列的pairs（大的在前）"""
        return sorted(pairs, key=lambda x: (x[0], x[1]), reverse=True)
    
    # 预验证期望结果
    candidate_pairs = [("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]
    expected_first = get_lex_order(candidate_pairs)[0]
    print(f"✓ 候选pairs字典序降序: {get_lex_order(candidate_pairs)}")
    print(f"✓ 期望优先合并: {expected_first}\n")
    
    # ========== Test 1: 基础验证 ("A","B") vs ("A","C") ==========
    print("=== Test 1: Simple tie-breaking ===")
    tokenizer1 = Tokenizer()

    words1 = [
        Word([65, 66]),  # "AB" -> pair: ("A", "B")
        Word([65, 67]),  # "AC" -> pair: ("A", "C")
    ]
    counts1 = [1, 1]  # 频率相同，触发平局
    
    tokenizer1.train_core_incremental(words1, counts1, vocab_size=259)
    
    # 验证：("A", "C") > ("A", "B")，应优先合并 ("A", "C")
    first_merge = list(tokenizer1.merges.keys())[0]
    print(f"First merge (token_ids): {first_merge}")
    # 65="A", 66="B", 67="C", 新token从256开始
    # 如果选中("A","C")，merge应为 (65, 67) -> 256
    assert first_merge[0] == 65 and first_merge[1] == 67, \
        f"Expected merge ('A','C')=(65,67), got {first_merge}"
    print("✓ Test 1 passed: ('A','C') selected over ('A','B')\n")
    
    # ========== Test 2: 验证 ("BA","A") > ("A","B") ==========
    tok2 = Tokenizer()
    # 先让 ("B","A") 高频生成 "BA"(id=256)
    tok2.train_core_incremental(
        [Word([66,65])]*10 + [Word([256,65]), Word([65,66])],  # BA×10, BAA, AB
        [1]*10 + [1, 1],
        vocab_size=259
    )
    # 第二轮: ("BA","A")=(256,65) vs ("A","B")=(65,66)
    # 字典序: ("BA","A") > ("A","B")，应选前者
    if len(tok2.merges) >= 2:
        second = list(tok2.merges.keys())[1]
        assert second == (256, 65), f"Expected ('BA','A'), got {second}"
        print("✓ Case2: ('BA','A') correctly selected")
   
if __name__ == "__main__":
    # test01()
    test_frequency_tie_breaking()
    
    