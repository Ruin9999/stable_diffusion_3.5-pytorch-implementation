import torch

def parse_parentheses(string: str):
    result = []
    current_item = ""
    nesting_level = 0
    for char in string:
        if char == "(":
            if nesting_level == 0:
                if current_item:
                    result.append(current_item)
                    current_item = "("
                else:
                    current_item = "("
            else:
                current_item += char
            nesting_level += 1
        elif char == ")":
            nesting_level -= 1
            if nesting_level == 0:
                result.append(current_item + ")")
                current_item = ""
            else:
                current_item += char
        else:
            current_item += char
    if current_item:
        result.append(current_item)
    return result


def token_weights(string: str, current_weight: float):
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if len(x) >= 2 and x[-1] == ")" and x[0] == "(":
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1
            if xx > 0:
                try:
                    weight = float(x[xx + 1 :])
                    x = x[:xx]
                except ValueError: # Co-pilot generated error
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out

def escape_important(string: str):
    string = string.replace("\\)", "\0\1")
    string = string.replace("\\(", "\0\2")
    return string


def unescape_important(string: str):
    string = string.replace("\0\1", ")")
    string = string.replace("\0\2", "(")
    return string

def modulate(x: torch.Tensor, shift: float, scale: float): # What the fuck are the dimensions here?
    if shift is None:
        shift = torch.zeros_list(scale)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def split_qkv(qkv, head_dim):
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 3, -1, head_dim).movedim(2, 0)
    return qkv[0], qkv[1], qkv[2]


class Tokenizer:
    """
    ClipTokenizer that defaults to clipL configuration
    """
    def __init__(
        self,
        tokenizer: None,
        max_length: int = 77,
        pad_with_end: bool = True,
        has_start_token: bool = True,
        pad_to_max_length: bool = True,
        min_length: int = None,
        extra_padding_token: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_with_end = pad_with_end
        self.has_start_token = has_start_token
        self.pad_to_max_length = pad_to_max_length
        self.min_length = min_length
        self.extra_padding_token = extra_padding_token
        self.max_word_length = 0 # Max number of sub-tokens to consider before a word is considered 'large'
        
        empty_tokens = self.tokenizer("")["input_ids"]
        
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty_tokens[0]
            self.end_token = empty_tokens[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty_tokens[0]
            
        vocab = self.tokenizer.get_vocab() # {token_string: token_id}
        self.inv_vocab = {v: k for k, v in vocab.items()} # {token_string: token_id} -> {token_id: token_string}
        
        
    def tokenize_with_weights(self, text: str, return_word_ids: bool = False):
        """
        Tokenize the text and produce weighted tokens
        For simplicity, each token defaults to weight=1.0 unless external logic changes it.
        
        Args:
            text: The input string to tokenize
            return_word_ids: if True, include a word index as the 3rd element of each tuple
            
        Returns:
            List of tuples of (token_id, weight) or (token_id, weight, word_id)
        """
        
        pad_token = self.end_token if self.pad_with_end else 0
        
        # 1) Extract weights segments
        text = escape_important(text)
        parsed_weights = token_weights(text, current_weight=1.0)
        
        # 2) Tokenize each segment (split by whitespace)
        tokens_per_word = self._tokenize_segments(parsed_weights)
        
        # 3) Convert tokens into batches
        batched_tokens = self._build_batches(tokens_per_word, pad_token)
        
        # 4) Optionally remove the word-index dimension
        if not return_word_ids:
            batched_tokens = [[(tid, w) for (tid, w, _) in batch] for batch in batched_tokens ]
            
        return batched_tokens
    
    def _tokenize_segments(self, parsed_weights):
        """
        Converts parsed segments of text & weights into a list of lists of (token_id, weight).
        Each sub-list corresponds to a single 'word' or 'segment'
        """
        
        tokens = []
        for (text_segment, weight) in parsed_weights:
            segment_text = unescape_important(text_segment).replace("\n", " ")
            words = [w for w in segment_text.split(" ") if w]
            
            for word in words:
                # The tokenizer outputs e.g. [start_token, ...tokens..., end_token]
                # We skip the start_token and end_token
                word_token_ids = self.tokenizer(word)["input_ids"][self.tokens_start: -1]
                tokens.append([(token_id, weight) for token_id in word_token_ids])
                
        return tokens
    
    def _build_batches(self, tokens_per_word, pad_token):
        """
        Takes a list of lists of (token_id, weight), and splits them into
        batches of length <= self.max_length, ensuring we have room of an end_token
        in each batch. We also handle extra padding and min_length
        """
        batched_tokens = []
        current_batch = []
        
        # Prepend separate start_token if we have it
        if self.start_token is not None:
            current_batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(current_batch)
        
        for word_index, t_group in enumerate(tokens_per_word):
            is_large = (len(t_group) > self.max_word_length)
            t_group = list(t_group)
            
            while t_group:
                # If adding this group would exceed max_length - 1, start a new batch
                if len(current_batch) + len(t_group) > (self.max_length -1):
                    # Break the batch
                    remaining_length = (self.max_length - 1) - len(current_batch)
                    
                    if is_large:
                        # Add partial tokens, and end token
                        to_add = t_group[:remaining_length]
                        current_batch.extend([(token_id, word, word_index + 1) for (token_id, word) in to_add])
                        current_batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    else:
                        current_batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            self._pad_batch(current_batch, pad_token)
                            
                    current_batch = []
                    if self.start_token is not None:
                        current_batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(current_batch)
                    
                else:
                    current_batch.extend([(token_id, word, word_index + 1) for (token_id, word) in t_group])
                    t_group = []
        
        # Handle extra padding token before the final end token
        self._pad_extra_token(current_batch)
        
        current_batch.append((self.end_token, 1.0, 0))
        
        if self.pad_to_max_length:
            self._pad_batch(current_batch, pad_token)
            
        if self.min_length is not None and len(current_batch) < self.min_length:
            missing = self.min_length - len(current_batch)
            current_batch.extend([(pad_token, 1.0, 0)] * missing)
            
        return batched_tokens
    
    def _pad_batch(self, current_batch, pad_token):
        """Pads the current batch to self.max_length with the given pad_token"""
        needed = self.max_length - len(current_batch)
        if needed > 0:
            current_batch.extend([(pad_token, 1.0, 0)] * needed)
            
    def _pad_extra_token(self, current_batch):
        """
        If extra_padding_token is set and we have a min_length,
        pad with that special token up to min_length - 1 (to leave room for the end_token)
        """
        if self.extra_padding_token is not None and self.min_length is not None:
            needed = self.min_length - len(current_batch) - 1
            if needed > 0:
                current_batch.extend([(self.extra_padding_token, 1.0, 0)] * needed)
                
    def untokenize(self, token_weight_pairs):
        """
        Given a list of (token_id, weight) or (token_id, weight, word_id),
        return a lsit of ((token_id, weight), token_string)
        """
        
        def _map_fn(tup):
            token_id = tup[0]
            return (tup, self.inv_vocab.get(token_id, f"<UNK_{token_id}>"))
            
        return list(map(_map_fn, token_weight_pairs))
    