# load packages
import time
import yaml
import argparse
from munch import Munch
import torch
import torchaudio
import librosa
import phonemizer
import soundfile as sf
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule


class InferenceModel():
    def __init__(self,
                 model,
                 sampler,
                 decoder_type,
                 n_mels=80,
                 n_fft=2048,
                 win_length=1200,
                 hop_length=300,
                 mean=-4,
                 std=4,
                 device='cuda'):
        # load phonemizer
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us',
            preserve_punctuation=True,
            with_stress=True)

        # text cleaner
        self.textclenaer = TextCleaner()

        # spectrogram
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length)
        self.mean, self.std = mean, std

        self.device = device
        self.model = model
        self.sampler = sampler
        self.decoder_type = decoder_type

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))
    
        return torch.cat([ref_s, ref_p], dim=1)

    def inference(self, text, ref_s, alpha = 0.0, beta = 0.5, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
    
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2) 

            s_pred = self.sampler(
                                noise = torch.randn((1, 256)).unsqueeze(1).to(self.device), 
                                embedding=bert_dur,
                                embedding_scale=embedding_scale,
                                features=ref_s, # reference from the same speaker as the embedding
                                num_steps=diffusion_steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en, 
                                         s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)


            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.decoder_type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.decoder_type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, 
                                     F0_pred,
                                     N_pred,
                                     ref.squeeze().unsqueeze(0))
    
        
        return out.squeeze()[..., :-50] # weird pulse at the end of the model, need to be fixed later

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # config
    config = yaml.safe_load(open(args.config))

    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    # load BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)

    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    # model checkpoint
    params_whole = torch.load(args.checkpoint_path, map_location='cpu')
    params = params_whole['net']

    # load model
    for key in model:
        if key in params:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict, strict=False)
    _ = [model[key].eval() for key in model]

    # diffuser
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )

    inference_model = InferenceModel(model, sampler, model_params.decoder.type, device=device)

    # synthesize a text
    if args.input_text:
        start = time.time()
        ref_s = inference_model.compute_style(args.cond_sample)
        wav = inference_model.inference(text, ref_s, diffusion_steps=10, embedding_scale=1)
        rtf = (time.time() - start) / (len(wav) / 24000)
        print(f"RTF = {rtf:5f}")
        sf.write('test_multispeaker.wav', wav.cpu().numpy(), 24000)
    else:
        os.makedirs(args.output_folder, exist_ok=True)
        with open(args.input_csv, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
        texts = [line.strip().split('|')[1] for line in lines]
        cond_paths = [line.strip().split('|')[0] for line in lines]
        for text, path in tqdm(zip(texts, cond_paths), total=len(texts)):
            ref_s = inference_model.compute_style(path)
            wav = inference_model.inference(text, ref_s, diffusion_steps=10, embedding_scale=1)
            out_path = os.path.join(args.output_folder, os.path.basename(path))
            sf.write(out_path, wav.cpu().numpy(), 24000)

    print(text)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to config.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='path to checkpoint.')
    parser.add_argument('--cond_sample', type=str, help='conditioning sample.')
    parser.add_argument('--input_text', default=None, help='text to be synthesized.')
    parser.add_argument('--input_csv', default=None, help='input csv to be synthesized: in this case the conditioning will be taken from the file itself.')
    parser.add_argument('--output_folder', default=None, help='location to save synthesized files.')
    args = parser.parse_args()
    main(args)