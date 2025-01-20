import argparse
import itertools
from phonemizer import phonemize
from tqdm import tqdm


def replace_symbols(symbols):
	symbols_new = symbols.replace('ɐ̃','A').replace('ʊ̃','W').replace('ũ','U').replace('õ','O').replace('ɲ̃','N').replace('ð','T')
	return symbols_new

def transcribe_sentence(sent, lang, replace=False):
    try:
        phonetic = phonemize(sent,
                         language=lang,
                         backend='espeak',
                         preserve_punctuation=True,
                         strip=True,
                         with_stress=True,
                         punctuation_marks=';:,.!?¡¿—…"«»“”.(){}[]')
    except:
         raise ValueError(f'\nCheck on this text: {sent}\n') 
    if replace:
        phonetic = replace_symbols(phonetic)
    return phonetic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='ptBR', type=str, help='language to be used: prBR|enUS (default: ptBR).')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size (default: 128)')
    parser.add_argument('--replace_symbols', default=False, help='whether to replace some symbols or not (default: False).')
    parser.add_argument('--file1', required=True, type=str, help='original train.txt')
    parser.add_argument('--file2', required=True, type=str, help='modified train.txt')

    args = parser.parse_args()
    file1 = args.file1
    file2 = args.file2
    accepted_lang = ['ptBR','enUS']
    if args.lang not in accepted_lang:
        raise ValueError(f'Language {args.lang} not supported yet.')
    lang = 'pt-br' if args.lang == 'ptBR'  else 'en-us'
    print(f'\n*LANGUAGE USED: {lang}\n')

    with open(file1, 'r', encoding='utf-8') as fp:
        file1_lines = fp.readlines()
    audiofiles = [t.strip().split('|')[0] for t in file1_lines]
    text = [t.strip().split('|')[1] for t in file1_lines]
    speaker_id = [t.strip().split('|')[2] for t in file1_lines]

    bs = args.batch_size
    nbatches = len(text) // bs
    nrest = len(text) - nbatches * bs

    graphemes = []
    phonemes = []
    transcriptions = []

    for i in tqdm(range(nbatches)):

        t = text[i*bs:(i+1)*bs]
        p = transcribe_sentence(t, lang, replace=args.replace_symbols)

        graphemes += list(itertools.chain(*t))
        phonemes += list(itertools.chain(*p))

        transcriptions += p

    i += 1

    if nrest:

        t = text[i*bs:i*bs+nrest]
        p = transcribe_sentence(t, lang, replace=args.replace_symbols)

        graphemes += list(itertools.chain(*t))
        phonemes += list(itertools.chain(*p))

        transcriptions += p

    assert len(transcriptions) == len(text), f'The number of transcriptions must match the number of texts.'

    grapheme_set = set(graphemes)
    phoneme_set = set(phonemes)
    grapheme_symbols = ''.join(list(grapheme_set))
    phoneme_symbols = ''.join(list(phoneme_set))

    print(f'\n\nGRAPHEME SET:\n {grapheme_set}')
    print(f'PHONEME SET:\n {phoneme_set}')
    print(f'GRAPHEME SYMBOLS:\n {grapheme_symbols}')
    print(f'PHONEME SYMBOLS:\n {phoneme_symbols}\n\n')

    with open(file2, 'w', encoding='utf-8') as fp:
        for f, t, p, s in zip(audiofiles, text, transcriptions, speaker_id):
            fp.write(f'{f}|{t}|{p}|{s}\n')

if __name__ == "__main__":
	main()
