import json
import os
import shutil  # zip
from pathlib import Path
from typing import Any, Tuple  # Callable, Dict, List, Optional

import gdown
import torch
import torchaudio



# Reference : https://github.com/morris-frank/nsynth-pytorch/blob/master/nsynth/data.py


class AKWDDataset(torch.utils.data.Dataset):

    # downladed from Adventure Kid Research & Technology (AKRT) website.
    # Link : https://www.adventurekid.se/akrt/waveforms/adventure-kid-waveforms/
    # deleated streo data. (200data)

    def __init__(
        self,
        root: str,
        download: bool = True,
        align_to_zero: bool = True,
        fix_boundary: bool = False,
        remove_dc: bool = True,
    ):
        super().__init__()
        self.root = root
        self.align_to_zero = align_to_zero
        self.fix_boundary = fix_boundary
        self.remove_dc = remove_dc

        if download:
            self.download()

        self.wave_paths = [str(p) for p in Path(self.root).glob("**/*.wav")]

        print(f"Loading AKWD data from {self.root}")
        print(f"\tFound {len(self)} samples.")
        assert len(self) > 0, f"No samples found in {self.root}"

    def __len__(self):
        return len(self.wave_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:

        # Args:
        #    index (int): Index
        # Returns:
        #    tuple: (image, target) where target is index of the target class.

        audio, sample_rate = torchaudio.load(self.wave_paths[idx])

        if self.align_to_zero:
            audio = self._align_start_to_zero(audio)
        if self.fix_boundary:
            audio = self._fix_boundary_discontinuity(audio)
        if self.remove_dc:
            audio = self._remove_dc(audio)

        # wave_paths[idx]
        with open(
            f"{self.root}/labels/{Path(self.wave_paths[idx]).stem}_analysis.json", "r"
        ) as fp:
            attrs = json.load(fp)

        # attrs['audio'] = waveform
        attrs["name"] = Path(self.wave_paths[idx]).name

        return audio, attrs

    @staticmethod
    def _align_start_to_zero(audio: torch.Tensor) -> torch.Tensor:
        
        # Remove DC offset
        audio = audio - audio.mean(dim=1, keepdim=True)

        # Zero-crossing index
        def find_shift(w: torch.Tensor) -> int:
            s = torch.sign(w)
            s[s == 0] = 1  # treat zeros as positive
            prod = s[:-1] * s[1:]
            crossings = torch.nonzero(prod < 0, as_tuple=False).flatten()

            # evaluate jump
            def jump_at(idx: int) -> torch.Tensor:
                # if we start at idx+1, boundary is between idx and idx+1
                a = w[idx]
                b = w[(idx + 1) % w.numel()]
                return (b - a).abs()

            best_idx = None
            best_jump = None

            # Prefer rising zero-crossings
            if crossings.numel() > 0:
                rising = crossings[(w[crossings] <= 0) & (w[crossings + 1] > 0)]
                candidates = rising if rising.numel() > 0 else crossings
                jumps = torch.stack([jump_at(int(i)) for i in candidates])
                min_pos = torch.argmin(jumps).item()
                best_idx = int(candidates[min_pos])
                best_jump = jumps[min_pos]

            # find globally minimal boundary jump
            if best_idx is None:
                jumps_all = (w - torch.roll(w, shifts=1)).abs()
                min_pos = torch.argmin(jumps_all).item()
                best_idx = min_pos - 1  # boundary before min_pos, so idx = min_pos-1
                best_jump = jumps_all[min_pos]

            return (best_idx + 1) % w.numel()

        shift = find_shift(audio[0])

        return torch.roll(audio, shifts=-shift, dims=1)

    @staticmethod
    def _fix_boundary_discontinuity(audio: torch.Tensor) -> torch.Tensor:

        # Boundary smoothing
        len = audio.shape[1]

        jump = audio[:, -1] - audio[:, 0] 
        ramp = torch.linspace(0.0, 1.0, steps=len, device=audio.device)
        return audio - jump * ramp

    @staticmethod
    def _remove_dc(audio: torch.Tensor) -> torch.Tensor:
        return audio - audio.mean(dim=1, keepdim=True)

    def download(self) -> None:
        if os.path.exists(self.root):
            print("Already downloaded")
            return
        else:
            cwd = os.getcwd()
            FILENAME = "AKWF_44k1_600s"
            DOWNLOAD_NAME = FILENAME + ".zip"
            # この辺のパスの与え方は修正したい.良い方法を考える
            UNPACK_PATH = os.path.join(cwd, "data")
            DOWNLOAD_PATH = os.path.join(UNPACK_PATH, DOWNLOAD_NAME)
            ID = "1Bpos6HJp6IHJYIkJ0rrXhydyeiA7gREO"  # ess_yeojohnson*DCO

            URL = "https://drive.google.com/uc?id=" + ID

            # if os.path.exists(os.path.join(cwd, DOWNLOAD_NAME)):
            if os.path.exists(DOWNLOAD_PATH):
                print("Already downloaded")
                return
            else:  # Datasetダウンロードとunzip
                gdown.download(URL, DOWNLOAD_PATH, quiet=True)  # 全部で4158個のファイル
                shutil.unpack_archive(DOWNLOAD_PATH, UNPACK_PATH)
                print("Download Complete")


if __name__ == "__main__":
    AKWDDataset("data/AKWF_44k1_600s/", download=True)
