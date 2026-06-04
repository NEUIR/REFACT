import re
from collections import Counter
from typing import List, Union

def detect_loop(text: str, min_sentence_len: int = 15, threshold: int = 3, debug: bool = False) -> bool:
    """
    Detects loop based on sentence repetition in the text.
    
    Args:
        text: The input text string.
        min_sentence_len: Minimum length of a sentence to be considered.
        threshold: If a sentence appears > threshold times, it's a loop.
        debug: If True, print debug information.
        
    Returns:
        True if a loop is detected.
    """
    if not text:
        return False

    # Check 1: Token/word level repetition (like "1 1 1 1 1")
    words = text.split()
    if debug:
        print(f"[Loop Debug] Words: {len(words)}")
    if len(words) > 10:  # Only check if enough words
        word_counts = Counter(words)
        if word_counts:
            most_common_word, word_freq = word_counts.most_common(1)[0]
            if debug:
                print(f"[Loop Debug] Token repetition detected: '{most_common_word}' appears {word_freq}/{len(words)} times ({word_freq/len(words):.1%})")
            word_ratio = word_freq / len(words)
            # If a single word/token appears more than 50% of the time, it's a loop
            if word_ratio > 0.5:
                if debug:
                    print(f"[Loop Debug] Token repetition detected: '{most_common_word}' appears {word_freq}/{len(words)} times ({word_ratio:.1%})")
                return True

    # Check 2: Pattern repetition (like "The final answer is\nThe answer is...")
    # Look for repeated patterns in consecutive lines
    lines = text.split('\n')
    if len(lines) > 5:
        # Check if many lines start with the same pattern
        line_starts = []
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                # Take first few words as pattern (up to 5 words or 30 chars)
                words_in_line = line.split()[:5]
                pattern = ' '.join(words_in_line)[:30]
                if len(pattern) > 3:  # Ignore very short patterns
                    line_starts.append(pattern)
        
        if len(line_starts) > 5:
            pattern_counts = Counter(line_starts)
            most_common_pattern, pattern_freq = pattern_counts.most_common(1)[0]
            pattern_ratio = pattern_freq / len(line_starts)
            
            # If the same pattern starts more than 40% of lines
            if pattern_ratio > 0.4 and pattern_freq > 3:
                if debug:
                    print(f"[Loop Debug] Pattern repetition detected: '{most_common_pattern}...' appears {pattern_freq}/{len(line_starts)} times ({pattern_ratio:.1%})")
                return True

    # Check 3: Sentence level repetition (original logic)
    # Split by common sentence delimiters (English and Chinese)
    sentences = re.split(r'[.?!;:\n\r。！？；：]+', text)
    
    # Filter empty or too short sentences and strip whitespace
    valid_sentences = [s.strip() for s in sentences if len(s.split()) >= min_sentence_len]
    
    if not valid_sentences:
        return False
    
    # Count frequencies in the whole text
    counts = Counter(valid_sentences)
    
    if not counts:
        return False
        
    # Check if any sentence exceeds the threshold
    most_common_sent, freq = counts.most_common(1)[0]
    
    if debug and freq > 2:  # Print if any repetition
        print(f"[Loop Debug] Most repeated sentence: '{most_common_sent[:50]}...' (freq={freq}, threshold={threshold})")
    
    if freq > threshold:
        return True

    # Check 4: N-gram diversity (Language Agnostic / Chinese Support)
    # Remove whitespace to handle concatenated words/Chinese
    text_no_space = "".join(text.split())
    
    # Only check if text is long enough to determine a loop
    if len(text_no_space) > 50:
        # Use trigrams (3-grams)
        n = 5
        ngrams = [text_no_space[i:i+n] for i in range(len(text_no_space)-n+1)]
        if len(ngrams) > 0:
            unique_ngrams = set(ngrams)
            diversity_ratio = len(unique_ngrams) / len(ngrams)
            
            if debug:
                print(f"[Loop Debug] N-gram diversity: {diversity_ratio:.1%} ({len(unique_ngrams)}/{len(ngrams)})")
            
            if diversity_ratio < 0.35:
                if debug:
                    print(f"[Loop Debug] Low n-gram diversity detected: {diversity_ratio:.1%}")
                return True

    return False


if __name__ == "__main__":
    text = """The answer should be "band" or "的音乐 organization band bandtype bandorganizationOrganizationOrganizationOrganizationOrganizationOrganizationORGorganOrganizationOrganizationOrganizationOrganizationORGbandOrganization组织Organization组织Organization组织组织音乐Organization组织组织组织组织组织 musicalOrganization组织Organization组织乐队music乐队band组织乐队乐队乐队音乐组织组织乐团mus组织组织组织组织乐队Band乐队组织乐队乐队乐队乐队音乐 band音乐组织组织音乐组织乐队乐队乐队乐队band组织组织组织乐队音乐组织band组织组织组织乐队 band组织组织组织和音乐乐队组织组织band组织组织音乐组织组织乐队组织组织组织组织band组织组织组织组织组织乐队组织组织音乐组织组织乐队组织组织音乐组织组织组织乐队组织组织组织乐队组织组织band组织组织组织 band组织组织音乐组织组织band组织组织组织组织组织乐队组织乐队组织组织音乐组织band组织组织组织音乐组织乐队组织组织组织组织组织组织乐队组织组织组织乐队组织组织组织组织组织组织乐队组织组织组织乐队组织组织组织组织组织乐队组织组织 organizations组织band组织组织组织组织组织组织音乐组织组织组织组织band组织组织组织组织组织音乐组织组织组织组织组织组织乐队组织组织组织组织 bands组织 band组织组织组织组织组织组织组织音乐组织乐队组织组织组织组织乐队组织组织组织组织组织音乐组织组织组织音乐组织组织组织组织乐队组织组织组织组织组织组织乐队组织音乐组织组织组织组织组织组织乐队组织组织组织组织组织乐队组织 音乐组织组织组织 musical组织组织组织乐队组织组织组织band组织组织组织组织音乐组织组织组织组织组织乐队组织组织盎band组织组织组织组织乐队组织组织音乐组织组织组织组织组织乐队组织组织组织组织组织组织组织组织组织组织组织组织组织组织音乐组织组织组织组织组织组织组织组织组织组织组织乐队组织组织组织组织组织组织成员组织组织组织组织组织组织乐队组织组织组织音乐组织组织组织组织组织组织组织组织组织组织组织组织 音乐组织组织组织组织组织组织音乐组织组织组织组织秩序组织组织组织组织 musical组织组织组织组织音乐组织组织组织组织组织组织组织音乐组织组织组织组织组织组织组织组织组织组织组织组织组织组织组织乐队组织组织组织组织组织乐队组织组织组织组织组织band组织组织类别乐队组织组织组织组织组织乐队组织乐队组织音乐组织组织组织组织组织 band组织组织组织组织音乐组织band组织组织组织组织音乐组织组织组织组织乐队组织组织押 band音乐组织组织组织组织组织组织组织组织组织组织音乐组织组织组织组织组织组织组织组织音乐组织组织组织组织组织组织乐队组织组织组织组织组织乐队组织组织组织组织组织组织组织团体band组织组织组织组织组织组织组织音乐组织组织组织组织乐队组织组织组织组织组织组织乐队组织组织组织组织组织组织组织组织组织成员乐队组织组织组织组织组织徐州组织组织组织组织组织乐队组织组织组织组织组织成员组织组织band组织组织组织组织组织组织音乐组织组织组织组织组织音乐组织组织组织组织组织组织团体 Band组织音乐组织组织组织组织组织组织乐队组织组织组织组织组织音乐组织组织组织组织组织组织音乐组织组织组织组织乐队组织组织组织组织组织组织组织组织组织团体组织组织音乐组织组织组织组织团体组织行政区组织组织组织组织组织组织band组织组织组织组织组织组织组织乐队组织组织组织乐队组织组织组织组织组织组织组织音乐组织组织组织组织组织组织组织组织团体组织组织组织组织音乐组织组织组织组织组织组织乐队组织组织组织成员组织组织组织组织音乐组织组织组织组织组织乐队组织组织组织组织组织乐队组织组织组织组织团体组织乐团组织组织组织组织乐队组织组织组织组织乐队组织组织组织成员组织 band组织组织组织组织组织组织band组织组织组织组织音乐组织band组织组织组织组织乐队组织组织组织乐队组织组织组织音乐组织组织组织band组织乐队组织组织组织乐队组织组织乐队组织band组织组织 band组织组织音乐组织Band组织组织组织组织组织乐队组织组织组织组织音乐组织组织乐队组织组织组织组织乐团组织组织组织组织组织组织乐队组织组织组织组织组织组织Band组织组织组织组织组织看见组织组织组织组织band组织组织组织组织band组织组织组织组织组织乐队音乐组织组织band组织组织组织组织音乐组织组织组织组织组织乐队组织组织组织组织Band组织组织组织组织组织组织乐队组织组织组织组织组织组织团体组织组织组织音乐组织组织组织组织组织组织组织组织团体组织组织乐队组织组织组织组织组织组织组织组织组织组织团体组织组织乐队组织组织组织组织组织组织组织组织的组织组织组织组织乐队组织组织组织成员组织组织组织组织组织组织乐队组织组织机构组织组织组织乐队组织组织组织为一体band组织组织组织组织组织组织组织乐队组织组织组织组织组织音乐组织组织组织组织组织乐队组织组织组织组织组织组织组织组织组织组织组织组织组织组织组织组织音乐组织组织组织组织组织组织组织组织组织组织团体组织组织团体组织组织组织成员组织组织组织组织组织学说组织组织组织组织组织组织组织组织组织组织组织组织组织 organization组织组织组织组织组织音乐组织组织组织团结组织音乐组织组织组织组织组织组织组织组织band组织组织组织组织组织组织音乐组织组织组织组织组织组织音乐组织音乐组织组织组织组织组织组织组织组织组织组织组织组织组织哈哈哈组织组织组织组织组织组织组织单位组织音乐组织组织组织组织组织组织组织组织机构band组织组织组织组织组织组织组织组织组织音乐组织组织成员组织组织组织组织音乐组织组织组织lane组织组织组织组织组织组织组织组织组织团体组织团体组织团体组织音乐组织组织组织组织实施组织组织组织组织组织组织组织组织组织成员组织组织组织组织组织团体组织音乐组织组织组织组织组织组织组织组织组织团体组织组织组织机构组织组织组织组织组织成员组织组织和组织组织组织组织编制组织成员组织组织组织组织组织组织音乐组织组织组织组织组织音乐组织组织信息组织组织组织器官组织组织安排组织组织组织组织组织组织组织组织组织音乐组织 organism组织组织组织组织组织组织组织组织团体组织麻将组织组织组织组织组织器官组织组织团体组织组织组织组织组织组织组织组织组织活跃组织组织组织组织组织组织组织组织组织单"""
    print(detect_loop(text, debug=True))