import sounddevice as sd
import torch, librosa, threading, pickle
from enhancer import Enhancer
import numpy as np
from torch.nn import functional as F
from torchaudio.transforms import Resample
from ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
from ddsp.core import upsample
import time
import _thread
from pynput import keyboard
# NOTE on M1 macs this requires the cffi package

flag_vc = False

def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = a * (fade_out ** 2) + b * (fade_in ** 2) + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    return result


class SvcDDSP:
    def __init__(self) -> None:
        self.model = None
        self.units_encoder = None
        self.encoder_type = None
        self.encoder_ckpt = None
        self.enhancer = None
        self.enhancer_type = None
        self.enhancer_ckpt = None

    def update_model(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using",self.device)

        # load ddsp model
        if self.model is None or self.model_path != model_path:
            self.model, self.args = load_model(model_path, device=self.device)
            self.model_path = model_path

            # load units encoder
            if self.units_encoder is None or self.args.data.encoder != self.encoder_type or self.args.data.encoder_ckpt != self.encoder_ckpt:
                if self.args.data.encoder == 'cnhubertsoftfish':
                    cnhubertsoft_gate = self.args.data.cnhubertsoft_gate
                else:
                    cnhubertsoft_gate = 10
                self.units_encoder = Units_Encoder(
                    self.args.data.encoder,
                    self.args.data.encoder_ckpt,
                    self.args.data.encoder_sample_rate,
                    self.args.data.encoder_hop_size,
                    cnhubertsoft_gate=cnhubertsoft_gate,
                    device=self.device)
                self.encoder_type = self.args.data.encoder
                self.encoder_ckpt = self.args.data.encoder_ckpt

        # load enhancer
        if self.enhancer is None or self.args.enhancer.type != self.enhancer_type or self.args.enhancer.ckpt != self.enhancer_ckpt:
            self.enhancer = Enhancer(self.args.enhancer.type, self.args.enhancer.ckpt, device=self.device)
            self.enhancer_type = self.args.enhancer.type
            self.enhancer_ckpt = self.args.enhancer.ckpt

    def infer(self,
              audio,
              sample_rate,
              spk_id=1,
              threhold=-45,
              pitch_adjust=0,
              use_spk_mix=False,
              spk_mix_dict=None,
              use_enhancer=True,
              enhancer_adaptive_key='auto',
              pitch_extractor_type='crepe',
              f0_min=50,
              f0_max=1100,
              safe_prefix_pad_length=0,
              ):
        
        # load input
        # audio, sample_rate = librosa.load(input_wav, sr=None, mono=True)
        hop_size = self.args.data.block_size * sample_rate / self.args.data.sampling_rate
    
        # safe front silence
        if safe_prefix_pad_length > 0.03:
            silence_front = safe_prefix_pad_length - 0.03
        else:
            silence_front = 0

        # extract f0
        pitch_extractor = F0_Extractor(
            pitch_extractor_type,
            sample_rate,
            hop_size,
            float(f0_min),
            float(f0_max))
        f0 = pitch_extractor.extract(audio, uv_interp=True, device=self.device, silence_front=silence_front)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(pitch_adjust) / 12)

        # extract volume
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)

        # extract units
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        units = self.units_encoder.encode(audio_t, sample_rate, hop_size)

        # spk_id or spk_mix_dict
        spk_id = torch.LongTensor(np.array([[spk_id]])).to(self.device)
        dictionary = None
        if use_spk_mix:
            dictionary = spk_mix_dict

        # forward and return the output
        with torch.no_grad():
            output, _, (s_h, s_n) = self.model(units, f0, volume, spk_id=spk_id, spk_mix_dict=dictionary)
            output *= mask
            if use_enhancer:
                output, output_sample_rate = self.enhancer.enhance(
                    output,
                    self.args.data.sampling_rate,
                    f0,
                    self.args.data.block_size,
                    adaptive_key=enhancer_adaptive_key,
                    silence_front=silence_front)
            else:
                output_sample_rate = self.args.data.sampling_rate

            output = output.squeeze()
            return output, output_sample_rate


class Config:
    def __init__(self) -> None:
        self.samplerate = 44100  # Hz
        self.block_time = 0.3  # s
        self.f_pitch_change: float = 0.0  # float(request_form.get("fPitchChange", 0))
        self.spk_id = 1
        self.spk_mix_dict = None  # {1:0.5, 2:0.5}
        self.use_vocoder_based_enhancer = True
        self.use_phase_vocoder = False
        self.checkpoint_path = ''
        self.threhold = -45
        self.crossfade_time = 0.04
        self.extra_time = 2.0
        self.select_pitch_extractor = 'rmvpe'  # ["parselmouth", "dio", "harvest", "crepe", "rmvpe", "fcpe"]
        self.use_spk_mix = False
        self.sounddevices = ['', '']

    def save(self, path):
        with open(path + '\\config.pkl', 'wb') as f:
            pickle.dump(vars(self), f)

    def load(self, path) -> bool:
        try:
            with open(path + '\\config.pkl', 'rb') as f:
                self.update(pickle.load(f))
            return True
        except:
            print('config.pkl does not exist')
            return False
    
    def update(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)


class App:
    def __init__(self) -> None:
        self.config = Config()
        self.block_frame = 0
        self.crossfade_frame = 0
        self.sola_search_frame = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.svc_model: SvcDDSP = SvcDDSP()
        self.fade_in_window: np.ndarray = None
        self.fade_out_window: np.ndarray = None
        self.input_wav: np.ndarray = None
        self.output_wav: np.ndarray = None
        self.sola_buffer: torch.Tensor = None
        self.f0_mode_list = ["parselmouth", "dio", "harvest", "crepe", "rmvpe", "fcpe"]
        self.f_safe_prefix_pad_length: float = 0.0
        self.resample_kernel = {}
        self.stream = None
        self.input_devices = None
        self.output_devices = None
        self.input_devices_indices = None 
        self.output_devices_indices = None
        self.update_devices()
        self.default_input_device = self.input_devices[self.input_devices_indices.index(sd.default.device[0])]
        self.default_output_device = self.output_devices[self.output_devices_indices.index(sd.default.device[1])]        

    def launch(self) -> None:
        print("press s to start/stop, q to quit")
        self.event_handler()

    def event_handler(self):
        global flag_vc
        while True:
            key = input()
            if key=="s" and not flag_vc:
                print("Starting voice conversion")        
                self.start_vc()
            elif key=="s" and flag_vc:
                print("Stopping voice conversion")        
                self.stop_stream()
                print("\npress s to start/stop, in/on to change in/out device, q to quit")
            elif key.startswith("i"):
                i = self.input_devices[int(key[1:].strip())]
                o = self.config.sounddevices[1]
                self.set_devices(i,o)
            elif key.startswith("o"):
                i = self.config.sounddevices[0]
                o = self.output_devices[int(key[1:].strip())]
                self.set_devices(i,o)
            elif key=="q":
                flag_vc = False
                exit()

        return 
        def on_key_press(key):
            global flag_vc
            # print("Key pressed: ", key)
            # so this is a bit of a quirk with pynput,
            # if an alpha-numeric key is pressed the key object will have an attribute
            # char which contains a string with the character, but it will only have
            # this attribute with alpha-numeric, so if a special key is pressed
            # this attribute will not be in the object.
            # so, we end up having to check if the attribute exists with the hasattr
            # function in python, and then check the character
            # here is that in action:
            if hasattr(key, "char"):
                if key.char == "s":
                    if not flag_vc:
                        print("Starting voice conversion")
                        self.start_vc()
                    else:
                        print("Stopping voice conversion")
                        self.stop_stream()
                elif key.char == "q":
                    flag_vc = False
                    keyboard_listener.stop()
                    keyboard_listener.join()
                    exit()
        keyboard_listener = keyboard.Listener(
            on_press=on_key_press)
        keyboard_listener.start()                            
        

    def __del__(self) -> None:
        self.stop_stream()

    def set_values(self, values):
        self.set_devices(values["sg_input_device"], values['sg_output_device'])
        self.config.sounddevices = [values["sg_input_device"], values['sg_output_device']]
        self.config.checkpoint_path = values['sg_model']
        self.config.spk_id = int(values['spk_id'])
        self.config.threhold = values['threhold']
        self.config.f_pitch_change = values['pitch']
        self.config.samplerate = int(values['samplerate'])
        self.config.block_time = float(values['block'])
        self.config.crossfade_time = float(values['crossfade'])
        self.config.extra_time = float(values['extra'])
        self.config.select_pitch_extractor = values['f0_mode']
        self.config.use_vocoder_based_enhancer = values['use_enhancer']
        self.config.use_phase_vocoder = values['use_phase_vocoder']
        self.config.use_spk_mix = values['spk_mix']
        self.block_frame = int(self.config.block_time * self.config.samplerate)
        self.crossfade_frame = int(self.config.crossfade_time * self.config.samplerate)
        self.sola_search_frame = int(0.01 * self.config.samplerate)
        self.last_delay_frame = int(0.02 * self.config.samplerate)
        self.extra_frame = int(self.config.extra_time * self.config.samplerate)
        self.input_frame = max(
            self.block_frame + self.crossfade_frame + self.sola_search_frame + 2 * self.last_delay_frame,
            self.block_frame + self.extra_frame)
        self.f_safe_prefix_pad_length = self.config.extra_time - self.config.crossfade_time - 0.01 - 0.02

    def start_vc(self):
        torch.cuda.empty_cache()
        self.input_wav = np.zeros(self.input_frame, dtype='float32')
        self.sola_buffer = torch.zeros(self.crossfade_frame, device=self.device)
        self.fade_in_window = torch.sin(
            np.pi * torch.arange(0, 1, 1 / self.crossfade_frame, device=self.device) / 2) ** 2
        self.fade_out_window = 1 - self.fade_in_window
        self.svc_model.update_model(self.config.checkpoint_path)
        self.start_stream()

    def start_stream(self):
        global flag_vc
        if not flag_vc:
            flag_vc = True
            self.stream = sd.Stream(
                channels=1,
                callback=self.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.config.samplerate,
                dtype="float32")
            self.stream.start()

    def stop_stream(self):
        global flag_vc
        if flag_vc:
            flag_vc = False
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                
    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        start_time = time.perf_counter()
        
        self.input_wav[:] = np.roll(self.input_wav, -self.block_frame)
        self.input_wav[-self.block_frame:] = librosa.to_mono(indata.T)

        # infer
        _audio, _model_sr = self.svc_model.infer(
            self.input_wav,
            self.config.samplerate,
            spk_id=self.config.spk_id,
            threhold=self.config.threhold,
            pitch_adjust=self.config.f_pitch_change,
            use_spk_mix=self.config.use_spk_mix,
            spk_mix_dict=self.config.spk_mix_dict,
            use_enhancer=self.config.use_vocoder_based_enhancer,
            pitch_extractor_type=self.config.select_pitch_extractor,
            safe_prefix_pad_length=self.f_safe_prefix_pad_length,
        )

        # debug sola
        '''
        _audio, _model_sr = self.input_wav, self.config.samplerate
        rs = int(np.random.uniform(-200,200))
        print('debug_random_shift: ' + str(rs))
        _audio = np.roll(_audio, rs)
        _audio = torch.from_numpy(_audio).to(self.device)
        '''

        if _model_sr != self.config.samplerate:
            key_str = str(_model_sr) + '_' + str(self.config.samplerate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(_model_sr, self.config.samplerate,
                                                         lowpass_filter_width=128).to(self.device)
            _audio = self.resample_kernel[key_str](_audio)
        temp_wav = _audio[
                   - self.block_frame - self.crossfade_frame - self.sola_search_frame - self.last_delay_frame: - self.last_delay_frame]

        # sola shift
        conv_input = temp_wav[None, None, : self.crossfade_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(conv_input ** 2, torch.ones(1, 1, self.crossfade_frame, device=self.device)) + 1e-8)
        sola_shift = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        temp_wav = temp_wav[sola_shift: sola_shift + self.block_frame + self.crossfade_frame]        

        # phase vocoder
        if self.config.use_phase_vocoder:
            temp_wav[: self.crossfade_frame] = phase_vocoder(
                self.sola_buffer,
                temp_wav[: self.crossfade_frame],
                self.fade_out_window,
                self.fade_in_window)
        else:
            temp_wav[: self.crossfade_frame] *= self.fade_in_window
            temp_wav[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window

        self.sola_buffer = temp_wav[- self.crossfade_frame:]

        #outdata[:] = temp_wav[: - self.crossfade_frame, None].repeat(1, 2).cpu().numpy()
        outdata[:] = temp_wav[: - self.crossfade_frame, None].cpu().numpy() # quito el repeat???
        end_time = time.perf_counter()
        if flag_vc:
            print(f'Infering, sola_shift: {int(sola_shift)}, infer_time: {(end_time - start_time)*1000}',end="\r")
""
    def update_devices(self):
        sd._terminate()
        sd._initialize()
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for hostapi in hostapis:
            for device_idx in hostapi["devices"]:
                devices[device_idx]["hostapi_name"] = hostapi["name"]
        self.input_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_input_channels"] > 0
        ]
        self.output_devices = [
            f"{d['name']} ({d['hostapi_name']})"
            for d in devices
            if d["max_output_channels"] > 0
        ]
        self.input_devices_indices = [d["index"] for d in devices if d["max_input_channels"] > 0]
        self.output_devices_indices = [
            d["index"] for d in devices if d["max_output_channels"] > 0
        ]

        print("Input devices:")
        print("-------------")
        for idx,dev in enumerate(self.input_devices):
            print(f"i{idx}: {dev}")
        print("Output devices:")
        print("--------------")
        for idx,dev in enumerate(self.output_devices):
            print(f"o{idx}: {dev}")

    def set_devices(self, input_device, output_device):
        sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
        sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]
        print("input device set to:" + str(sd.default.device[0]) + ":" + str(input_device))
        print("output device set to:" + str(sd.default.device[1]) + ":" + str(output_device))
    
if __name__ == "__main__":
    app = App()
    # app.config.load("exp/combsum/config.yaml") # `config.yaml` path model_30000.pt

    values = dict()
    values['sg_input_device'] = 'MacBook Air Microphone (Core Audio)'
    values['sg_output_device'] = 'MacBook Air Speakers (Core Audio)'
    values['sg_model'] = "exp/combsum/model_10000.pt" # app.config.checkpoint_path
    values['spk_id'] = app.config.spk_id
    values['threhold'] = app.config.threhold
    values['pitch'] = app.config.f_pitch_change
    values['samplerate'] = app.config.samplerate
    values['block'] = app.config.block_time
    values['crossfade'] = app.config.crossfade_time
    values['extra'] = app.config.extra_time
    values['f0_mode'] = app.config.select_pitch_extractor
    values['use_enhancer'] = app.config.use_vocoder_based_enhancer
    values['use_phase_vocoder'] = app.config.use_phase_vocoder
    values['spk_mix'] = app.config.use_spk_mix

    app.set_values(values)

    app.update_devices()

    app.launch()

