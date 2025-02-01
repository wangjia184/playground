# Adapted from https://github.com/TMElyralab/MuseTalk/blob/main/musetalk/whisper/audio2feature.py

import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisper import load_model
import numpy as np
import torch



class Audio2Feature:
    def __init__(
        self,
        model_path="checkpoints/whisper/tiny.pt",
        device=None,
        audio_embeds_cache_dir=None,
        num_frames=16,
    ):
        self.model = load_model(model_path, device)
        self.audio_embeds_cache_dir = audio_embeds_cache_dir
        self.num_frames = num_frames
        self.embedding_dim = self.model.dims.n_audio_state

    def get_sliced_feature(self, feature_array, vid_idx, audio_feat_length=[2, 2], fps=25):
        """
        Get sliced features based on a given index
        :param feature_array:
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return:
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        center_idx = int(vid_idx * 50 / fps)
        left_idx = center_idx - audio_feat_length[0] * 2
        right_idx = center_idx + (audio_feat_length[1] + 1) * 2

        for idx in range(left_idx, right_idx):
            idx = max(0, idx)
            idx = min(length - 1, idx)
            x = feature_array[idx]
            selected_feature.append(x)
            selected_idx.append(idx)

        selected_feature = torch.cat(selected_feature, dim=0)
        selected_feature = selected_feature.reshape(-1, self.embedding_dim)  # 50*384
        return selected_feature, selected_idx

    def get_sliced_feature_sparse(self, feature_array, vid_idx, audio_feat_length=[2, 2], fps=25):
        """
        Get sliced features based on a given index
        :param feature_array:
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return:
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        for dt in range(-audio_feat_length[0], audio_feat_length[1] + 1):
            left_idx = int((vid_idx + dt) * 50 / fps)
            if left_idx < 1 or left_idx > length - 1:
                left_idx = max(0, left_idx)
                left_idx = min(length - 1, left_idx)

                x = feature_array[left_idx]
                x = x[np.newaxis, :, :]
                x = np.repeat(x, 2, axis=0)
                selected_feature.append(x)
                selected_idx.append(left_idx)
                selected_idx.append(left_idx)
            else:
                x = feature_array[left_idx - 1 : left_idx + 1]
                selected_feature.append(x)
                selected_idx.append(left_idx - 1)
                selected_idx.append(left_idx)
        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, self.embedding_dim)  # 50*384
        selected_feature = torch.from_numpy(selected_feature)
        return selected_feature, selected_idx

    def feature2chunks(self, feature_array, fps, audio_feat_length=[2, 2]):
        whisper_chunks = []
        whisper_idx_multiplier = 50.0 / fps
        i = 0
        print(f"video in {fps} FPS, audio idx in 50FPS")

        while True:
            start_idx = int(i * whisper_idx_multiplier)
            selected_feature, selected_idx = self.get_sliced_feature(
                feature_array=feature_array, vid_idx=i, audio_feat_length=audio_feat_length, fps=fps
            )
            # print(f"i:{i},selected_idx {selected_idx}")
            whisper_chunks.append(selected_feature)
            i += 1
            if start_idx > len(feature_array):
                break

        return whisper_chunks

    def _audio2feat(self, audio_path: str):
        result = self.model.transcribe(audio_path)
        
        embed_list = []
        for emb in result["segments"]:
            encoder_embeddings = emb["encoder_embeddings"]                # shape = (1, 5, T, 384)
            encoder_embeddings = encoder_embeddings.transpose(0, 2, 1, 3) # shape = (1, T, 5, 384)
            encoder_embeddings = encoder_embeddings.squeeze(0)            # shape = (T, 5, 384)
            # start_idx/end_idx values are absolute positions in the original mel-spectrogram
            start_idx = int(emb["start"])
            end_idx = int(emb["end"])
            # end_idx - start_idx = the number of original mel frames
            # Whisper encoder uses convolutional layers with a stride of 2, halving the time dimension
            emb_end_idx = int((end_idx - start_idx) / 2)
            embed_list.append(encoder_embeddings[:emb_end_idx])
        concatenated_array = torch.from_numpy(np.concatenate(embed_list, axis=0))
        return concatenated_array

    def audio2feat(self, audio_path):
        if self.audio_embeds_cache_dir == "" or self.audio_embeds_cache_dir is None:
            return self._audio2feat(audio_path)

        audio_embeds_cache_path = os.path.join(self.audio_embeds_cache_dir, os.path.basename(audio_path) + ".pt")

        if os.path.isfile(audio_embeds_cache_path):
            try:
                audio_feat = torch.load(audio_embeds_cache_path)
            except Exception as e:
                print(f"{type(e).__name__} - {e} - {audio_embeds_cache_path}")
                os.remove(audio_embeds_cache_path)
                audio_feat = self._audio2feat(audio_path)
                torch.save(audio_feat, audio_embeds_cache_path)
        else:
            audio_feat = self._audio2feat(audio_path)
            torch.save(audio_feat, audio_embeds_cache_path)

        return audio_feat

    def crop_overlap_audio_window(self, audio_feat, start_index):
        selected_feature_list = []
        for i in range(start_index, start_index + self.num_frames):
            selected_feature, selected_idx = self.get_sliced_feature(
                feature_array=audio_feat, vid_idx=i, audio_feat_length=[2, 2], fps=25
            )
            selected_feature_list.append(selected_feature)
        mel_overlap = torch.stack(selected_feature_list)
        return mel_overlap


if __name__ == "__main__":
    import ffmpeg

    """
    Key Concept: Temporal Context Window
        Video FPS: 25 frames per second (1 frame = 40 ms).
        Audio "FPS": 50 indices per second (1 index = 20 ms).
        Mapping: Each video frame is associated with a window of 10 audio indices (spanning 200 ms) to provide temporal context, not just 2 indices (40 ms).
    """
    current_directory = os.path.dirname(__file__)
    model_path = f"{os.path.dirname(os.path.dirname(current_directory))}/models/tiny.pt"
    print("Loading model from", model_path)
    audio_encoder = Audio2Feature(model_path=model_path)
    audio_path = f"{current_directory}/assets/demo1_audio.wav"
    print("Decoding audio file", audio_path)

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        pcm_buffer, _ = (
            ffmpeg.input(audio_path, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000) # Sample rate must be 16k Hz
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    audio_samples = np.frombuffer(pcm_buffer, np.int16).flatten().astype(np.float32) / 32768.0
    print("audio_samples.shape ", audio_samples.shape)
    

    array = audio_encoder.audio2feat(audio_samples)
    print("array.shape =", array.shape) # Shape = [T, 5, 384] where for each indice it represents 20ms
    fps = 25
    whisper_idx_multiplier = 50.0 / fps

    i = 0
    print(f"video in {fps} FPS, audio idx in 50FPS")
    while True:
        start_idx = int(i * whisper_idx_multiplier)
        selected_feature, selected_idx = audio_encoder.get_sliced_feature(
            feature_array=array, vid_idx=i, audio_feat_length=[2, 2], fps=fps
        )
        print(f"video idx {i},\t audio idx {selected_idx},\t shape {selected_feature.shape}")
        i += 1
        if start_idx > len(array):
            break
