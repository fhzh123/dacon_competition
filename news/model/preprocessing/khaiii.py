# Import modules
from tqdm import tqdm
from khaiii import KhaiiiApi
from collections import Counter

class khaiii_encoder:
    def __init__(self, args, verbose=True):
        # Token setting
        self.pad_idx = args.pad_idx
        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.unk_idx = args.unk_idx
        self.sep_idx = args.sep_idx

        # Path setting
        self.save_path = args.save_path

        # Training setting
        self.vocab_size = args.vocab_size
        self.verbose = verbose

        # API setting
        self.api = KhaiiiApi()

    def parsing_sentence(self, text_list):

        print('Khaiii word counting...')

        if self.verbose:
            p_bar = tqdm(text_list)
        else:
            p_bar = text_list

        parsing_list = list()

        for text in p_bar:
            parsing_text = list()
            for word_ in self.api.analyze(text):
                for word in word_.morphs:
                    parsing_text.extend(word.lex)
            parsing_list.append(parsing_text)
        
        return parsing_list

    def counter_update(self, title_parsed_list, content_parsed_list):

        print('Khaiii word2id setting...')

        lex_counter = Counter()

        for text in title_parsed_list + content_parsed_list:
            lex_counter.update(text)

        word2id = ['<pad>', '<bos>', '<eos>', '<unk>','[SEP]']
        if len(lex_counter.values()) >= self.vocab_size:
            min_count = sorted(list(lex_counter.values()), reverse=True)[self.vocab_size] + 1
        else:
            min_count = 2
        word2id.extend([w for w, freq in lex_counter.items() if freq >= min_count and w != '[SEP]'])
        word2id = {w: i for i, w in enumerate(word2id)}

        self.word2id = word2id

        return word2id

    def encode_sentence(self, parsing_title_list, parsing_content_list):

        print('Khaiii encoding...')

        if self.verbose:
            p_bar_title = tqdm(parsing_title_list)
            p_bar_content = tqdm(parsing_content_list)
        else:
            p_bar_title = parsing_title_list
            p_bar_content = parsing_content_list

        title_indices = [
            [self.bos_idx] + [self.word2id.get(w, self.unk_idx) for w in title] + [self.eos_idx] \
                for title in p_bar_title
        ]
        content_indices = [
            [self.bos_idx] + [self.word2id.get(w, self.unk_idx) for w in title] + [self.eos_idx] \
                for title in p_bar_content
        ]
        total_indices =[
            title_[:-1] + [self.sep_idx] + content_[1:] \
                for title_, content_ in zip(title_indices, content_indices)
        ]

        return total_indices, title_indices, content_indices