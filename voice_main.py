from soni_translate.logging_setup import logger
import torch
import gc
import numpy as np
import os
import shutil
import warnings
import threading
from tqdm import tqdm
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.audio import load_audio
import soundfile as sf
import edge_tts
import asyncio
from soni_translate.utils import (
    remove_directory_contents,
    create_directories,
    write_chunked,
)
from scipy import signal
from time import time as ttime
import faiss
from vci_pipeline import VC, change_rms, highpass_filter_b, highpass_filter_a
import librosa

warnings.filterwarnings("ignore")


class Config:
    def __init__(self, only_cpu=False):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        (
            self.x_pad,
            self.x_query,
            self.x_center,
            self.x_max
        ) = self.device_config(only_cpu)

    def device_config(self, only_cpu) -> tuple:
        if torch.cuda.is_available() and not only_cpu:
            device_index = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(device_index)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info(
                    "16/10 Series GPUs and P40 excel "
                    "in single-precision tasks."
                )
                self.is_half = False
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(device_index).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
        elif torch.backends.mps.is_available() and not only_cpu:
            logger.info("Supported N-card not found, using MPS for inference")
            self.device = "mps"
        else:
            logger.info("No supported N-card found, using CPU for inference")
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = os.cpu_count()

        if self.is_half:
            # 6GB VRAM configuration
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5GB VRAM configuration
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        logger.info(
            f"Config: Device is {self.device}, "
            f"half precision is {self.is_half}"
        )

        return x_pad, x_query, x_center, x_max


BASE_DOWNLOAD_LINK = "https://huggingface.co/r3gm/sonitranslate_voice_models/resolve/main/"
BASE_MODELS = [
    "hubert_base.pt",
    "rmvpe.pt"
]
BASE_DIR = "."


def load_hu_bert(config):
    from fairseq import checkpoint_utils
    from soni_translate.utils import download_manager

    for id_model in BASE_MODELS:
        download_manager(
            os.path.join(BASE_DOWNLOAD_LINK, id_model), BASE_DIR
        )

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()

    return hubert_model


def load_trained_model(model_path, config):

    if not model_path:
        raise ValueError("No model found")

    logger.info("Loading %s" % model_path)
    model_checkpoint = torch.load(model_path, map_location="cpu")
    target_sample_rate = model_checkpoint["config"][-1]
    model_checkpoint["config"][-3] = model_checkpoint["weight"]["emb_g.weight"].shape[0]  # n_spk
    has_pitch_features = model_checkpoint.get("f0", 1)
    if has_pitch_features == 0:
        # protect to 0.5 need?
        pass

    version = model_checkpoint.get("version", "v1")
    if version == "v1":
        if has_pitch_features == 1:
            generator_network = SynthesizerTrnMs256NSFsid(
                *model_checkpoint["config"], is_half=config.is_half
            )
        else:
            generator_network = SynthesizerTrnMs256NSFsid_nono(*model_checkpoint["config"])
    elif version == "v2":
        if has_pitch_features == 1:
            generator_network = SynthesizerTrnMs768NSFsid(
                *model_checkpoint["config"], is_half=config.is_half
            )
        else:
            generator_network = SynthesizerTrnMs768NSFsid_nono(*model_checkpoint["config"])
    del generator_network.enc_q

    generator_network.load_state_dict(model_checkpoint["weight"], strict=False)
    generator_network.eval().to(config.device)

    if config.is_half:
        generator_network = generator_network.half()
    else:
        generator_network = generator_network.float()

    vc = VC(target_sample_rate, config)
    num_speakers = model_checkpoint["config"][-3]

    return num_speakers, target_sample_rate, generator_network, vc, model_checkpoint, version


class ClassVoices:
    def __init__(self, only_cpu=False):
        self.model_config = {}
        self.config = None
        self.only_cpu = only_cpu

    def apply_conf(
        self,
        tag="base_model",
        file_model="",
        pitch_algo="pm",
        pitch_lvl=0,
        file_index="",
        index_influence=0.66,
        respiration_median_filtering=3,
        envelope_ratio=0.25,
        consonant_breath_protection=0.33,
        resample_sr=0,
        file_pitch_algo="",
    ):

        if not file_model:
            raise ValueError("Model not found")

        if file_index is None:
            file_index = ""

        if file_pitch_algo is None:
            file_pitch_algo = ""

        if not self.config:
            self.config = Config(self.only_cpu)
            self.hu_bert_model = None
            self.model_pitch_estimator = None

        self.model_config[tag] = {
            "file_model": file_model,
            "pitch_algo": pitch_algo,
            "pitch_lvl": pitch_lvl,  # no decimal
            "file_index": file_index,
            "index_influence": index_influence,
            "respiration_median_filtering": respiration_median_filtering,
            "envelope_ratio": envelope_ratio,
            "consonant_breath_protection": consonant_breath_protection,
            "resample_sr": resample_sr,
            "file_pitch_algo": file_pitch_algo,
        }
        return f"CONFIGURATION APPLIED FOR {tag}: {file_model}"

    def infer(
        self,
        task_id,
        params,
        # load model
        num_speakers,
        target_sample_rate,
        generator_network,
        pipe,
        model_checkpoint,
        version,
        has_pitch_features,
        # load index
        index_rate,
        index,
        big_npy,
        # load f0 file
        input_pitch_override,
        # audio file
        input_audio_path,
        overwrite,
    ):

        f0_method = params["pitch_algo"]
        f0_up_key = params["pitch_lvl"]
        filter_radius = params["respiration_median_filtering"]
        resample_sr = params["resample_sr"]
        rms_mix_rate = params["envelope_ratio"]
        protect = params["consonant_breath_protection"]

        if not os.path.exists(input_audio_path):
            raise ValueError(
                "The audio file was not found or is not "
                f"a valid file: {input_audio_path}"
            )

        f0_up_key = int(f0_up_key)

        audio = load_audio(input_audio_path, 16000)

        # Normalize audio
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max

        times = [0, 0, 0]

        # filters audio signal, pads it, computes sliding window sums,
        # and extracts optimized time indices
        audio = signal.filtfilt(highpass_filter_b, highpass_filter_a, audio)
        audio_pad = np.pad(
            audio, (pipe.window // 2, pipe.window // 2), mode="reflect"
        )
        opt_ts = []
        if audio_pad.shape[0] > pipe.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(pipe.window):
                audio_sum += audio_pad[i:i - pipe.window]
            for interval_point in range(pipe.t_center, audio.shape[0], pipe.t_center):
                opt_ts.append(
                    interval_point
                    - pipe.t_query
                    + np.where(
                        np.abs(audio_sum[interval_point - pipe.t_query: interval_point + pipe.t_query])
                        == np.abs(audio_sum[interval_point - pipe.t_query: interval_point + pipe.t_query]).min()
                    )[0][0]
                )

        audio_segment_start = 0
        audio_opt = []
        segment_split_point = None
        time_pitch_start = ttime()

        sid_value = 0
        speaker_id_tensor = torch.tensor(sid_value, device=pipe.device).unsqueeze(0).long()

        # Pads audio symmetrically, calculates length divided by window size.
        audio_pad = np.pad(audio, (pipe.t_pad, pipe.t_pad), mode="reflect")
        pitch_frame_length = audio_pad.shape[0] // pipe.window

        # Estimates pitch from audio signal
        pitch, pitch_float_values = None, None
        if has_pitch_features == 1:
            pitch, pitch_float_values = pipe.get_f0(
                input_audio_path,
                audio_pad,
                pitch_frame_length,
                f0_up_key,
                f0_method,
                filter_radius,
                input_pitch_override,
            )
            pitch = pitch[:pitch_frame_length]
            pitch_float_values = pitch_float_values[:pitch_frame_length]
            if pipe.device == "mps":
                pitch_float_values = pitch_float_values.astype(np.float32)
            pitch = torch.tensor(
                pitch, device=pipe.device
            ).unsqueeze(0).long()
            pitch_float_values = torch.tensor(
                pitch_float_values, device=pipe.device
            ).unsqueeze(0).float()

        time_pitch_end = ttime()
        times[1] += time_pitch_end - time_pitch_start
        for segment_split_point in opt_ts:
            segment_split_point = segment_split_point // pipe.window * pipe.window
            if has_pitch_features == 1:
                pitch_slice = pitch[
                    :, audio_segment_start // pipe.window: (segment_split_point + pipe.t_pad2) // pipe.window
                ]
                pitchf_slice = pitch_float_values[
                    :, audio_segment_start // pipe.window: (segment_split_point + pipe.t_pad2) // pipe.window
                ]
            else:
                pitch_slice = None
                pitchf_slice = None

            audio_slice = audio_pad[audio_segment_start:segment_split_point + pipe.t_pad2 + pipe.window]
            audio_opt.append(
                pipe.vc(
                    self.hu_bert_model,
                    generator_network,
                    speaker_id_tensor,
                    audio_slice,
                    pitch_slice,
                    pitchf_slice,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[pipe.t_pad_tgt:-pipe.t_pad_tgt]
            )
            audio_segment_start = segment_split_point

        pitch_end_slice = pitch[
            :, segment_split_point // pipe.window:
        ] if segment_split_point is not None else pitch
        pitchf_end_slice = pitch_float_values[
            :, segment_split_point // pipe.window:
        ] if segment_split_point is not None else pitch_float_values

        audio_opt.append(
            pipe.vc(
                self.hu_bert_model,
                generator_network,
                speaker_id_tensor,
                audio_pad[segment_split_point:],
                pitch_end_slice,
                pitchf_end_slice,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )[pipe.t_pad_tgt:-pipe.t_pad_tgt]
        )

        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(
                audio, 16000, audio_opt, target_sample_rate, rms_mix_rate
            )
        if resample_sr >= 16000 and target_sample_rate != resample_sr:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=target_sample_rate, target_sr=resample_sr
            )
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitch_float_values, speaker_id_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if target_sample_rate != resample_sr >= 16000:
            final_sr = resample_sr
        else:
            final_sr = target_sample_rate

        """
        "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            times[0],
            times[1],
            times[2],
        ), (final_sr, audio_opt)

        """

        if overwrite:
            output_audio_path = input_audio_path  # Overwrite
        else:
            basename = os.path.basename(input_audio_path)
            dirname = os.path.dirname(input_audio_path)

            new_basename = basename.split(
                '.')[0] + "_edited." + basename.split('.')[-1]
            new_path = os.path.join(dirname, new_basename)
            logger.info(str(new_path))

            output_audio_path = new_path

        # Save file
        write_chunked(
            file=output_audio_path,
            samplerate=final_sr,
            data=audio_opt,
            format="ogg",
            subtype="vorbis",
        )

        self.model_config[task_id]["result"].append(output_audio_path)
        self.output_list.append(output_audio_path)

    def make_test(
        self,
        tts_text,
        tts_voice,
        model_path,
        index_path,
        transpose,
        f0_method,
    ):

        folder_test = "test"
        tag = "test_edge"
        tts_file = "test/test.wav"
        tts_edited = "test/test_edited.wav"

        create_directories(folder_test)
        remove_directory_contents(folder_test)

        if "SET_LIMIT" == os.getenv("DEMO"):
            if len(tts_text) > 60:
                tts_text = tts_text[:60]
                logger.warning("DEMO; limit to 60 characters")

        try:
            asyncio.run(edge_tts.Communicate(
                tts_text, "-".join(tts_voice.split('-')[:-1])
            ).save(tts_file))
        except Exception as e:
            raise ValueError(
                "No audio was received. Please change the "
                f"tts voice for {tts_voice}. Error: {str(e)}"
            )

        shutil.copy(tts_file, tts_edited)

        self.apply_conf(
            tag=tag,
            file_model=model_path,
            pitch_algo=f0_method,
            pitch_lvl=transpose,
            file_index=index_path,
            index_influence=0.66,
            respiration_median_filtering=3,
            envelope_ratio=0.25,
            consonant_breath_protection=0.33,
        )

        self(
            audio_files=tts_edited,
            tag_list=tag,
            overwrite=True
        )

        return tts_edited, tts_file

    def run_threads(self, threads):
        # Start threads
        for thread in threads:
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        gc.collect()
        torch.cuda.empty_cache()

    def unload_models(self):
        self.hu_bert_model = None
        self.model_pitch_estimator = None
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(
        self,
        audio_files=[],
        tag_list=[],
        overwrite=False,
        parallel_workers=1,
    ):
        logger.info(f"Parallel workers: {str(parallel_workers)}")

        self.output_list = []

        if not self.model_config:
            raise ValueError("No model has been configured for inference")

        if isinstance(audio_files, str):
            audio_files = [audio_files]
        if isinstance(tag_list, str):
            tag_list = [tag_list]

        if not audio_files:
            raise ValueError("No audio found to convert")
        if not tag_list:
            tag_list = [list(self.model_config.keys())[-1]] * len(audio_files)

        if len(audio_files) > len(tag_list):
            logger.info("Extend tag list to match audio files")
            extend_number = len(audio_files) - len(tag_list)
            tag_list.extend([tag_list[0]] * extend_number)

        if len(audio_files) < len(tag_list):
            logger.info("Cut list tags")
            tag_list = tag_list[:len(audio_files)]

        tag_file_pairs = list(zip(tag_list, audio_files))
        sorted_tag_file = sorted(tag_file_pairs, key=lambda x: x[0])

        # Base params
        if not self.hu_bert_model:
            self.hu_bert_model = load_hu_bert(self.config)

        cache_params = None
        threads = []
        progress_bar = tqdm(total=len(tag_list), desc="Progress")
        for i, (id_tag, input_audio_path) in enumerate(sorted_tag_file):

            if id_tag not in self.model_config.keys():
                logger.info(
                    f"No configured model for {id_tag} with {input_audio_path}"
                )
                continue

            if (
                len(threads) >= parallel_workers
                or cache_params != id_tag
                and cache_params is not None
            ):

                self.run_threads(threads)
                progress_bar.update(len(threads))

                threads = []

            if cache_params != id_tag:

                self.model_config[id_tag]["result"] = []

                # Unload previous
                (
                    num_speakers,
                    target_sample_rate,
                    generator_network,
                    pipe,
                    model_checkpoint,
                    version,
                    has_pitch_features,
                    index_rate,
                    index,
                    big_npy,
                    input_pitch_override,
                ) = [None] * 11
                gc.collect()
                torch.cuda.empty_cache()

                # Model params
                params = self.model_config[id_tag]

                model_path = params["file_model"]
                f0_method = params["pitch_algo"]
                file_index = params["file_index"]
                index_rate = params["index_influence"]
                f0_file = params["file_pitch_algo"]

                # Load model
                (
                    num_speakers,
                    target_sample_rate,
                    generator_network,
                    pipe,
                    model_checkpoint,
                    version
                ) = load_trained_model(model_path, self.config)
                has_pitch_features = model_checkpoint.get("f0", 1)  # pitch data

                # Load index
                if os.path.exists(file_index) and index_rate != 0:
                    try:
                        index = faiss.read_index(file_index)
                        big_npy = index.reconstruct_n(0, index.ntotal)
                    except Exception as error:
                        logger.error(f"Index: {str(error)}")
                        index_rate = 0
                        index = big_npy = None
                else:
                    logger.warning("File index not found")
                    index_rate = 0
                    index = big_npy = None

                # Load f0 file
                input_pitch_override = None
                if os.path.exists(f0_file):
                    try:
                        with open(f0_file, "r") as f:
                            lines = f.read().strip("\n").split("\n")
                        input_pitch_override = []
                        for line in lines:
                            input_pitch_override.append([float(i) for i in line.split(",")])
                        input_pitch_override = np.array(input_pitch_override, dtype="float32")
                    except Exception as error:
                        logger.error(f"f0 file: {str(error)}")

                if "rmvpe" in f0_method:
                    if not self.model_pitch_estimator:
                        from lib.rmvpe import RMVPE

                        logger.info("Loading vocal pitch estimator model")
                        self.model_pitch_estimator = RMVPE(
                            "rmvpe.pt",
                            is_half=self.config.is_half,
                            device=self.config.device
                        )

                    pipe.model_rmvpe = self.model_pitch_estimator

                cache_params = id_tag

            # self.infer(
            #     id_tag,
            #     params,
            #     # load model
            #     num_speakers,
            #     target_sample_rate,
            #     generator_network,
            #     pipe,
            #     model_checkpoint,
            #     version,
            #     has_pitch_features,
            #     # load index
            #     index_rate,
            #     index,
            #     big_npy,
            #     # load f0 file
            #     input_pitch_override,
            #     # output file
            #     input_audio_path,
            #     overwrite,
            # )

            thread = threading.Thread(
                target=self.infer,
                args=(
                    id_tag,
                    params,
                    # loaded model
                    num_speakers,
                    target_sample_rate,
                    generator_network,
                    pipe,
                    model_checkpoint,
                    version,
                    has_pitch_features,
                    # loaded index
                    index_rate,
                    index,
                    big_npy,
                    # loaded f0 file
                    input_pitch_override,
                    # audio file
                    input_audio_path,
                    overwrite,
                )
            )

            threads.append(thread)

        # Run last
        if threads:
            self.run_threads(threads)

        progress_bar.update(len(threads))
        progress_bar.close()

        final_result = []
        valid_tags = set(tag_list)
        for tag in valid_tags:
            if (
                tag in self.model_config.keys()
                and "result" in self.model_config[tag].keys()
            ):
                final_result.extend(self.model_config[tag]["result"])

        return final_result
