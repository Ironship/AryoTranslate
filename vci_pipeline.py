import numpy as np, parselmouth, torch, pdb, sys
from time import time as ttime
import torch.nn.functional as F
import scipy.signal as signal
import pyworld, os, traceback, faiss, librosa, torchcrepe
from scipy import signal
from functools import lru_cache
from soni_translate.logging_setup import logger

now_dir = os.getcwd()
sys.path.append(now_dir)

highpass_filter_b, highpass_filter_a = signal.butter(N=5, Wn=48, btype="high", fs=16000)

audio_path_to_waveform_cache = {}


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = audio_path_to_waveform_cache[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):  # 1 is the input audio, 2 is the output audio, rate is the proportion of 2
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # one dot every half second
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class VC(object):
    def __init__(self, target_sample_rate, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000  # hubert input sampling rate
        self.window = 160  # points per frame
        self.t_pad = self.sr * self.x_pad  # Pad time before and after each bar
        self.t_pad_tgt = target_sample_rate * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # Query time before and after the cut point
        self.t_center = self.sr * self.x_center  # Query point cut position
        self.t_max = self.sr * self.x_max  # Query-free duration threshold
        self.device = config.device

    def get_f0(
        self,
        input_audio_path,
        audio_input,
        pitch_frame_length,
        pitch_shift_semitones,
        f0_method,
        filter_radius,
        input_pitch_override=None,
    ):
        global audio_path_to_waveform_cache
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(audio_input, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (pitch_frame_length - len(f0) + 1) // 2
            if pad_size > 0 or pitch_frame_length - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, pitch_frame_length - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            audio_path_to_waveform_cache[input_audio_path] = audio_input.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            model = "full"
            # Pick a batch size that doesn't cause memory errors on your gpu
            batch_size = 512
            # Compute pitch using first gpu
            audio = torch.tensor(np.copy(audio_input))[None].float()
            f0, pd = torchcrepe.predict(
                audio,
                self.sr,
                self.window,
                f0_min,
                f0_max,
                model,
                batch_size=batch_size,
                device=self.device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
        elif "rmvpe" in f0_method:
            if hasattr(self, "model_rmvpe") == False:
                from lib.rmvpe import RMVPE

                logger.info("Loading vocal pitch estimator model")
                self.model_rmvpe = RMVPE(
                    "rmvpe.pt", is_half=self.is_half, device=self.device
                )
            thred = 0.03
            if "+" in f0_method:
                f0 = self.model_rmvpe.pitch_based_audio_inference(audio_input, thred, f0_min, f0_max)
            else:
                f0 = self.model_rmvpe.infer_from_audio(audio_input, thred)

        f0 *= pow(2, pitch_shift_semitones / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0_frames_per_second = self.sr // self.window  # f0 points per second
        if input_pitch_override is not None:
            delta_t = np.round(
                (input_pitch_override[:, 0].max() - input_pitch_override[:, 0].min()) * f0_frames_per_second + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), input_pitch_override[:, 0] * 100, input_pitch_override[:, 1]
            )
            shape = f0[self.x_pad * f0_frames_per_second : self.x_pad * f0_frames_per_second + len(replace_f0)].shape[0]
            f0[self.x_pad * f0_frames_per_second : self.x_pad * f0_frames_per_second + len(replace_f0)] = replace_f0[
                :shape
            ]
        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0_backup = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        try:
            f0_coarse = np.rint(f0_mel).astype(np.int)
        except: # noqa
            f0_coarse = np.rint(f0_mel).astype(int)
        return f0_coarse, f0_backup  # 1-0

    def vc(
        self,
        model,
        generator_network,
        speaker_id,
        input_audio_chunk,
        pitch,
        pitch_float_values,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):  # ,file_index,file_big_npy
        feats = torch.from_numpy(input_audio_chunk)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        time_feature_start = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch != None and pitch_float_values != None:
            feats0 = feats.clone()
        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
            and index_rate != 0
        ):
            feature_embeddings = feats[0].cpu().numpy()
            if self.is_half:
                feature_embeddings = feature_embeddings.astype("float32")

            # _, I = index.search(feature_embeddings, 1)
            # feature_embeddings = big_npy[I.squeeze()]

            score, neighbor_indices = index.search(feature_embeddings, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            feature_embeddings = np.sum(big_npy[neighbor_indices] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                feature_embeddings = feature_embeddings.astype("float16")
            feats = (
                torch.from_numpy(feature_embeddings).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch != None and pitch_float_values != None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        time_vc_start = ttime()
        pitch_frame_length = input_audio_chunk.shape[0] // self.window
        if feats.shape[1] < pitch_frame_length:
            pitch_frame_length = feats.shape[1]
            if pitch != None and pitch_float_values != None:
                pitch = pitch[:, :pitch_frame_length]
                pitch_float_values = pitch_float_values[:, :pitch_frame_length]

        if protect < 0.5 and pitch != None and pitch_float_values != None:
            pitchff = pitch_float_values.clone()
            pitchff[pitch_float_values > 0] = 1
            pitchff[pitch_float_values < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        pitch_frame_length = torch.tensor([pitch_frame_length], device=self.device).long()
        with torch.no_grad():
            if pitch != None and pitch_float_values != None:
                output_audio_chunk = (
                    (generator_network.infer(feats, pitch_frame_length, pitch, pitch_float_values, speaker_id)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                output_audio_chunk = (
                    (generator_network.infer(feats, pitch_frame_length, speaker_id)[0][0, 0]).data.cpu().float().numpy()
                )
        del feats, pitch_frame_length, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time_vc_end = ttime()
        times[0] += time_vc_start - time_feature_start
        times[2] += time_vc_end - time_vc_start
        return output_audio_chunk

    def pipeline(
        self,
        model,
        generator_network,
        speaker_id,
        audio,
        input_audio_path,
        times,
        pitch_shift_semitones,
        f0_method,
        file_index,
        # file_big_npy,
        index_rate,
        if_f0,
        filter_radius,
        target_sample_rate,
        resample_sample_rate,
        rms_mix_rate,
        version,
        protect,
        f0_file=None,
    ):
        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and os.path.exists(file_index) == True
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                # big_npy = np.load(file_big_npy)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None
            logger.warning("File index Not found, set None")

        audio = signal.filtfilt(highpass_filter_b, highpass_filter_a, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for interval_point in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    interval_point
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[interval_point - self.t_query : interval_point + self.t_query])
                        == np.abs(audio_sum[interval_point - self.t_query : interval_point + self.t_query]).min()
                    )[0][0]
                )
        audio_segment_start = 0
        audio_opt = []
        segment_split_point = None
        time_pitch_start = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        pitch_frame_length = audio_pad.shape[0] // self.window
        input_pitch_override = None
        if hasattr(f0_file, "name") == True:
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                input_pitch_override = []
                for line in lines:
                    input_pitch_override.append([float(i) for i in line.split(",")])
                input_pitch_override = np.array(input_pitch_override, dtype="float32")
            except:
                traceback.print_exc()
        speaker_id = torch.tensor(speaker_id, device=self.device).unsqueeze(0).long()
        pitch, pitch_float_values = None, None
        if if_f0 == 1:
            pitch, pitch_float_values = self.get_f0(
                input_audio_path,
                audio_pad,
                pitch_frame_length,
                pitch_shift_semitones,
                f0_method,
                filter_radius,
                input_pitch_override,
            )
            pitch = pitch[:pitch_frame_length]
            pitch_float_values = pitch_float_values[:pitch_frame_length]
            if self.device == "mps":
                pitch_float_values = pitch_float_values.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitch_float_values = torch.tensor(pitch_float_values, device=self.device).unsqueeze(0).float()
        time_pitch_end = ttime()
        times[1] += time_pitch_end - time_pitch_start
        for segment_split_point in opt_ts:
            segment_split_point = segment_split_point // self.window * self.window
            if if_f0 == 1:
                audio_opt.append(
                    self.vc(
                        model,
                        generator_network,
                        speaker_id,
                        audio_pad[audio_segment_start : segment_split_point + self.t_pad2 + self.window],
                        pitch[:, audio_segment_start // self.window : (segment_split_point + self.t_pad2) // self.window],
                        pitch_float_values[:, audio_segment_start // self.window : (segment_split_point + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        generator_network,
                        speaker_id,
                        audio_pad[audio_segment_start : segment_split_point + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            audio_segment_start = segment_split_point
        if if_f0 == 1:
            audio_opt.append(
                self.vc(
                    model,
                    generator_network,
                    speaker_id,
                    audio_pad[segment_split_point:],
                    pitch[:, segment_split_point // self.window :] if segment_split_point is not None else pitch,
                    pitch_float_values[:, segment_split_point // self.window :] if segment_split_point is not None else pitch_float_values,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    generator_network,
                    speaker_id,
                    audio_pad[segment_split_point:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, target_sample_rate, rms_mix_rate)
        if resample_sample_rate >= 16000 and target_sample_rate != resample_sample_rate:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=target_sample_rate, target_sr=resample_sample_rate
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitch_float_values, speaker_id
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
