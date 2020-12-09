import sentencepiece as spm

class sentencepiece_encoder:
    def __init__(self, args):
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
        self.sentencepiece_tokenizer = args.sentencepiece_tokenizer

    def training(self, train_title_list, train_content_list):
        # 1) Make Korean text to train vocab
        with open(f'{self.save_path}/text.txt', 'w') as f:
            for text in train_title_list + train_content_list:
                f.write(f'{text}\n')

        # 2) SentencePiece model training
        spm.SentencePieceProcessor()
        spm.SentencePieceTrainer.Train(
            f'--input={self.save_path}/text.txt --model_prefix={self.save_path}/m_text '
            f'--vocab_size={self.vocab_size} --character_coverage=0.9995 '
            f'--model_type={self.sentencepiece_tokenizer} --split_by_whitespace=true '
            f'--pad_id={self.pad_idx} --unk_id={self.unk_idx} '
            f'--bos_id={self.bos_idx} --eos_id={self.eos_idx} --user_defined_symbols=[SEP]')

        # 3) Korean vocabulrary setting
        vocab_list = list()
        with open(f'{self.save_path}/m_text.vocab') as f:
            for line in f:
                vocab_list.append(line[:-1].split('\t')[0])
        word2id_spm = {w: i for i, w in enumerate(vocab_list)}
        
        return word2id_spm

    def encoding(self, title_list, content_list):
        # 1) SentencePiece model load
        spm_ = spm.SentencePieceProcessor()
        spm_.Load(f"{self.save_path}/m_text.model")

        # 2) Tokenizing
        title_indices = [[self.bos_idx] + spm_.EncodeAsIds(title) + [self.eos_idx] \
                        for title in title_list]
        content_indices = [[self.bos_idx] + spm_.EncodeAsIds(content) + [self.eos_idx] \
                        for content in content_list]
        total_indices = [[self.bos_idx] + spm_.EncodeAsIds(title) + [self.sep_idx] + \
                        spm_.EncodeAsIds(content) + [self.eos_idx] \
                        for title, content in zip(title_list, content_list)]

        return total_indices, title_indices, content_indices