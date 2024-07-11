'''
Write the BasicTokenizer class, with the following three core functions:

def train(self, text, vocab_size, verbose=False)
def encode(self, text)
def decode(self, ids)
'''

class BasicTokenizer():

    def __init__(self) -> None:
        self.merges = {}
        self.vocab = {}

    def _get_bigram_freq(self, text):
        stats = {}
        for c1, c2 in zip(text, text[1:]):
            stats[(c1, c2)] = stats.get((c1, c2), 0) + 1
        return stats

    def _replace_bigram(self, text, bigram_to_replace, replacement_token):
        processed_text = []
        i = 0
        while i < len(text):
            if i < len(text) - 1 and bigram_to_replace == (text[i], text[i+1]):
                processed_text.append(replacement_token)
                i += 2
            else:
                processed_text.append(text[i])
                i += 1
        return processed_text

    def train(self, text, vocab_size, verbose=False):
        # convert text into utf-8 encoded bytes
        utf_8_encoded_text = list(text.encode('utf-8'))
        # for (vocab_size - 256) turns:
        for i in range(vocab_size - 256):
            # get most freq repeating char bigram in text
            stats = self._get_bigram_freq(utf_8_encoded_text)
            if stats == {}:
                break
            
            most_freq_bigram = max(stats, key=stats.get)

            # mint a new token for the winning bigram
            self.merges[most_freq_bigram] = i + 256

            # replace every instance of winning bigram in text with newly minted token
            processed_text = self._replace_bigram(utf_8_encoded_text, most_freq_bigram, self.merges[most_freq_bigram])
            print(f"merged {most_freq_bigram} | {stats[most_freq_bigram]} -> {len(processed_text)}")
            # make the newly compressed text as original text
            utf_8_encoded_text = processed_text
        
        # after minting new tokens for all the merges we make a vocab for easy decoding later
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (c1, c2), v in self.merges.items():
            self.vocab[v] = self.vocab[c1] + self.vocab[c2]

    def encode(self, text):
        # encode to utf-8 bytes
        encoded_utf8_text = list(text.encode('utf-8'))
        # print(encoded_utf8_text)
        while len(encoded_utf8_text) > 1:
            # get all bigrams usign the get_stats we wrote before
            stats = self._get_bigram_freq(encoded_utf8_text)
            
            bigram_merge_idxs = {bigram: self.merges.get(bigram, float('inf')) for bigram in stats}
            lowest_bigram = min(bigram_merge_idxs, key=bigram_merge_idxs.get)
            if self.merges.get(lowest_bigram) is None:
                break
            
            encoded_utf8_text = self._replace_bigram(encoded_utf8_text, lowest_bigram, self.merges[lowest_bigram])
            # print(encoded_utf8_text)
        return encoded_utf8_text

    def decode(self, token_list):
        return b"".join(self.vocab[i] for i in token_list).decode('utf-8')



with open("taylorswift.txt", "r") as file:
    text = file.read()

tokenizer = BasicTokenizer()
tokenizer.train(text, 300)
# print(list(text.encode("utf-8")))
print(tokenizer.merges)
print("-"*50)
print(text)
print(tokenizer.decode(tokenizer.encode(text)))
print(tokenizer.decode(tokenizer.encode(text)) == text)
