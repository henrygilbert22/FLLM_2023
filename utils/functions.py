import math
import numpy as np
from collections import Counter


def entropy(text: str) -> float:
    counter = Counter(text)
    n = len(text)
    entropy = 0.0
    for freq in counter.values():
        p = freq / n
        entropy += p * math.log2(p)
    return -entropy

def compression_ratio(original_size: float, compressed_size: float) -> float:
    return 1 - (compressed_size / original_size)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def edit_distance(str1: str, str2: str) -> int:
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
        
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                
    return dp[m][n]

