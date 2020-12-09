from konlpy.tag import Mecab, Okt, Hannanum, Kkma, Komoran
from collections import Counter

class konlpy_encoder:
    def __init__(self, args, type='mecab'):
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

        # API setting
        if type.lower() == 'mecab':
            self.api = Mecab()
        if type.lower() == 'okt':
            self.api = Okt()
        if type.lower() == 'hannanum':
            self.api = Hannanum()
        if type.lower() == 'kkma':
            self.api = Kkma()
        if type.lower() == 'komoran':
            self.api = Komoran()

    def parsing_sentence(self, text_list):

        parsing_list = list()

        for text in text_list:
            parsing_list.append(self.api.morphs(text))
        
        return parsing_list

    def counter_update(self, title_parsed_list, content_parsed_list):

        lex_counter = Counter()

        for text in title_parsed_list + content_parsed_list:
            lex_counter.update(text)

        word2id = ['<pad>', '<bos>', '<eos>', '<unk>','[SEP]']
        min_count = sorted(list(lex_counter.values()), reverse=True)[self.vocab_size] + 1
        word2id.extend([w for w, freq in lex_counter.items() if freq >= min_count and w != '[SEP]'])
        word2id = {w: i for i, w in enumerate(word2id)}
        
        self.word2id = word2id

        return word2id

    def encode_sentence(self, parsing_title_list, parsing_content_list):

        title_indices = [
            [self.bos_idx] + [self.word2id.get(w, self.unk_idx) for w in title] + [self.eos_idx] \
                for title in parsing_title_list
        ]
        content_indices = [
            [self.bos_idx] + [self.word2id.get(w, self.unk_idx) for w in title] + [self.eos_idx] \
                for title in parsing_content_list
        ]
        total_indices =[
            title_[:-1] + [self.sep_idx] + content_[1:] \
                for title_, content_ in zip(title_indices, content_indices)
        ]

        return total_indices, title_indices, content_indices