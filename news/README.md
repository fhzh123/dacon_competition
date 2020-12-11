dacon link: https://dacon.io/competitions/official/235658/overview/

## 12월 7일 02:00
아이디어:
- sklearn, 등을 통해 단어 빈도 기반의 모델 생성 필요
- 일정 confidence 이하 문장에 대해서 단어 빈도 기반 모델 접목
변경점:
- 단어 빈도 기반의 모델 -> 약 93-94%대의 성능

## 12월 8일 01:30
아이디어:
- 일정 confidence 이하 문장에 대해서 단어 빈도 기반 머신러닝 모델 접목
- RNN에 src_total 투입 / 성능 비교 필요
변경점:

## 12월 8일 15:00
아이디어:
- Knowledge distillation 필요

## 12월 9일 15:00
아이디어:
- Khaiii, mecab, 등 KoNLPy도입
- Embedding에 noise추가

## 12월 10일 02:00
아이디어:
- Embedding에 noise추가
변경점:
- KoNLPy, Khaiii 추가
- 현재 수정중 (아직 안돌아감, 각 parser마다 인풋의 크기가 달라서 concat이 안됨)

## 12월 12일 02:00
아이디어:
- 일부 토큰만 사용하는 것이 아닌 전체 토큰에 pooling 진행
- LSTM도 진행
- Word2Vec 학습
변경점:
- 이전에 수정한 KoNLPy, Khaiii에 대해서 학습 가능