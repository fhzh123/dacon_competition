# Import modules
from tqdm import tqdm
from collections import Counter
from konlpy.tag import Mecab, Okt, Hannanum, Kkma, Komoran

class konlpy_encoder:
    def __init__(self, args, konlpy_type='mecab', verbose=True):
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
        if konlpy_type.lower() == 'mecab':
            self.api = Mecab()
        elif konlpy_type.lower() == 'okt':
            self.api = Okt()
        elif konlpy_type.lower() == 'hannanum':
            self.api = Hannanum()
        elif konlpy_type.lower() == 'kkma':
            self.api = Kkma()
        elif konlpy_type.lower() == 'komoran':
            self.api = Komoran()
        else:
            raise Exception('Not supported konlpy parser')

    def parsing_sentence(self, text_list):

        parsing_list = list()

        for text in text_list:
            parsing_list.append(self.api.morphs(text))
        
        return parsing_list

    def counter_update(self, title_parsed_list, content_parsed_list):

        print('KoNLPy word counting...')

        lex_counter = Counter()

        if self.verbose:
            p_bar = tqdm(title_parsed_list + content_parsed_list)
        else:
            p_bar = title_parsed_list + content_parsed_list
            
        for text in p_bar:
            lex_counter.update(text)

        word2id = ['<pad>', '<bos>', '<eos>', '<unk>','[SEP]']
        min_count = sorted(list(lex_counter.values()), reverse=True)[self.vocab_size] + 1
        word2id.extend([w for w, freq in lex_counter.items() if freq >= min_count and w != '[SEP]'])
        word2id = {w: i for i, w in enumerate(word2id)}
        
        self.word2id = word2id

        return word2id

    def encode_sentence(self, parsing_title_list, parsing_content_list):

        print('KoNLPy encoding...')

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
            [self.bos_idx] + [self.word2id.get(w, self.unk_idx) for w in content] + [self.eos_idx] \
                for content in p_bar_content
        ]
        total_indices =[
            title_[:-1] + [self.sep_idx] + content_[1:] \
                for title_, content_ in zip(title_indices, content_indices)
        ]

        return total_indices, title_indices, content_indices