import librosa
import numpy as np
import scipy
import threading

from gfm_iaif import gfm_iaif

import sounddevice as sd

sd.default.latency = 'low'

unused = """
import soundfile as sf
from functools import partial
from utils import h1h2, weighted_log_mag_mse_loss
from tract_proxy import VocalTractProxy
from optimize import TractControlsOptimizer
"""

current_frame = 0
class Resynth:
    def __init__(self, framelength=512, hoplength=256, fmin=70, fmax = 500, ncilinders = 44, fs=44100):
        self.framelength = framelength
        self.hoplength = hoplength
        self.fmin = fmin
        self.fmax = fmax
        self.ncilinders = ncilinders
        self.fs = fs
        self.process_blocks = 1
        self.stream = None
        self.prev_frames = 2
        self.prev_audio_orig = None
        self.nknobs = 22
        self.params = {
            'vt_shifts': [0,0,0],
            'glottis_shifts': 1,
            'tilt_factor': 1,
            'knobs': [0]*self.nknobs
        }
        self.input_devices = []
        self.output_devices = []
        self.input_devices_indices = []
        self.output_devices_indices = []   
        self.state = {
            'distorsion_tract': 0,
            'distorsion_glottis': 0
        }
        self.update_devices()     

    def __del__(self) -> None:
        self.stop_stream()

    def start_stream(self):
        self.prev_audio_orig = np.zeros( self.prev_frames * self.framelength, dtype="float32")        

        self.stream = sd.Stream(
            channels=1,
            callback=self.audio_callback,
            blocksize=self.process_blocks*self.framelength, # TODO HARDCODED Buffer length !!!!
            samplerate=self.fs,
            dtype="float32")
        print("Starting stream")
        self.stream.start()

    def stop_stream(self):
        if self.stream is not None:
            print("Stopping stream")
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_latency(self):       
        if self.stream is not None:
            L = self.stream.latency
            return L[0]+L[1]
        else:
            return -1
    
    def update_parameter(self,param,value):
        self.params.update({param: value})

    def play_audio(self,audio_file, samplerate, actually_play=True):  # convert an audio file and then play it
        global current_frame      
        cur_framerate = self.fs
        self.fs = samplerate

        input_wav = librosa.to_mono(audio_file.T)

        self.prev_audio_orig = np.zeros( self.prev_frames * self.framelength, dtype="float32")  
        # read parameters
        if "vt_shifts" in self.params:
            vt_shifts = self.params['vt_shifts']
        else:
            vt_shifts = []

        if "glottis_shifts" in self.params:
            glottis_shifts = self.params['glottis_shifts']  
        else:
            glottis_shifts = 1
            
        if "tilt_factor" in self.params:
            tilt_factor = self.params['tilt_factor']  
        else:
            tilt_factor = 1
        # 
        # call audio processing!!!
        #
        output_wav = self.process(
            input_wav,
            tract_shifts_per=vt_shifts,
            glottis_shift=glottis_shifts,
            tilt_factor=tilt_factor,
        )     
        if actually_play: 
            sd.play(output_wav, samplerate=self.fs)
        self.fs = cur_framerate
        return
    
    def stream_audio(self,audio_file,samplerate):  # stream with a thread
        global current_frame      
        current_frame = 0
        cur_framerate = self.fs
        self.fs = samplerate

        self.prev_audio_orig = np.zeros( self.prev_frames * self.framelength, dtype="float32")        

        def callback(outdata, frames, time, status):
            global current_frame
            chunksize = min(audio_file.shape[0] - current_frame, frames)
            
            indata = audio_file[current_frame:current_frame + chunksize]
            if chunksize < frames: 
                outdata[chunksize:] = 0
                raise sd.CallbackStop()

            _outdata = np.empty_like(outdata[:chunksize])
            self.audio_callback(indata, _outdata, frames, time, status)
            outdata[:chunksize] = _outdata #audio_file[current_frame:current_frame + chunksize]

            current_frame += chunksize

        event = threading.Event()
        stream = sd.OutputStream(channels=1,
                                callback=callback,
                                blocksize=self.process_blocks*self.framelength, 
                                samplerate=self.fs,
                                dtype="float32",
                        finished_callback=event.set)
        with stream:
            event.wait()
        self.fs = cur_framerate
        return

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
    
        if indata.shape[0]==0:
            outdata[:]=0
            return
        
        input_wav = np.concatenate( (self.prev_audio_orig, librosa.to_mono(indata.T)))

        # read parameters
        if "vt_shifts" in self.params:
            vt_shifts = self.params['vt_shifts']
        else:
            vt_shifts = []

        if "glottis_shifts" in self.params:
            glottis_shifts = self.params['glottis_shifts']  
        else:
            glottis_shifts = 0
            
        if "tilt_factor" in self.params:
            tilt_factor = self.params['tilt_factor']  
        else:
            tilt_factor = None
        # 
        # call audio processing!!!
        #
        output_wav = self.process(
            input_wav,
            tract_shifts_per=vt_shifts,
            glottis_shift=glottis_shifts,
            tilt_factor=tilt_factor,
        )

        #  keep previous 2 frames
        self.prev_audio_orig = input_wav[-2*self.framelength:] 
        # remove previous ending from signal and extra frame at beginning
        output_wav = output_wav[self.framelength:-self.framelength]
        
        # TODO find a  way to check channels!!!!
        output_wav = output_wav.reshape(outdata.shape)

        #outdata[:] = output_wav
        outdata[:] = output_wav


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

    def set_devices(self, input_device, output_device):
        sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
        sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]

    def get_devices(self):
        return self.input_devices, self.output_devices
    
    def process(self, audio_input, tract_shifts_per=None, glottis_shift=None, tilt_factor=None, return_parameters=False):
        
        # esta pirula de aqui es porque estoy probando ajustarme a un framelength variable
        # para dejar que sounddevice controle la latencia. No funciona todavia
        inner_framelength = self.framelength #min(self.framelength,audio_input.shape[0])
        inner_hoplength = self.hoplength #max(8,inner_framelength/default_hopratio)
        #
        #
        # first decompose in frames
        input_frames = librosa.util.frame(audio_input, 
                                          frame_length=inner_framelength, 
                                          hop_length=inner_hoplength)
        nframes = input_frames.shape[1]

        # get the LPC coefficients using the GFM-IAIB method
        tract_coeffs, glottis_coeffs, lip_coeffs = self.estimate_coeffs(input_frames)
        old_tract_coeffs = tract_coeffs.copy()
        old_glottis_coeffs = glottis_coeffs.copy()

        # remove vocal tract filter, isolate glottis signal
        # glottis_signal = self.filter_frames(input_frames, tract_coeffs, np.ones([1])) 
        # remove glottis filter, isolate excitation signal
        #excitation_signal = self.filter_frames(glottis_signal, glottis_coeffs, np.ones([1]))

        # NEW METHOD TO
        # calculate resonant frequency and quality factor from glottis poles
        glottis_poles = np.apply_along_axis(np.roots, 0, glottis_coeffs.astype(np.complex128))        
        glottis_poles = np.where( np.isclose(glottis_poles.imag,0), glottis_poles.real, glottis_poles)
        glottis_poles = np.apply_along_axis(lambda x: x[x.imag.argsort()], 0, glottis_poles)
        glottis_poles_real = glottis_poles[1,:].real
        glottis_poles_pos = glottis_poles[2,:]
        glottis_freqs = np.angle(glottis_poles_pos)
        # glottis_qs = - 1 / np.tan(np.angle(glottis_poles) / 2)

        # calculate resonant frequencies of vocal tract
        assert tract_coeffs.shape[1] % 2 == 1

        # this is how it should be done 
        # but roots inside apply_along_axis has numerical errors?
        # tract_poles = np.apply_along_axis(np.roots, 1, tract_coeffs.astype(np.complex128)).astype(np.complex128)
        # #tract_poles = np.array([np.roots(tract_coeffs[i,:]) for i in range(tract_coeffs.shape[0])])
        # tract_poles_pos = np.apply_along_axis(lambda x: x[np.angle(x).argsort()][len(x) // 2:], 1, tract_poles.astype(np.complex128))
        # tract_freqs = np.angle(tract_poles_pos)
        # tract_qs = - 1 / np.tan(np.angle(tract_poles) / 2)

        # so for now we do it like this
        # Vocal tract roots
        tract_poles = np.zeros((self.ncilinders, nframes),dtype=np.complex128)
        tract_poles_pos = np.empty((int(self.ncilinders/2), nframes),dtype=np.complex128)
        tract_freqs = np.empty((int(self.ncilinders/2),nframes))
        mask = np.empty((nframes))
        for n in range(nframes):     
            poles = np.roots(tract_coeffs[:,n])
            phase_poles = np.array([r for r in poles if np.imag(r) > 0])
            if phase_poles.shape[0] == self.ncilinders/2:
                tract_poles[:,n] = poles.copy()
                tract_poles_pos[:,n] = phase_poles.copy()    
                freqs = np.arctan2(phase_poles.imag, phase_poles.real) 
                idx_sort = freqs.argsort() 
                tract_freqs[:,n] = freqs[idx_sort].real
                tract_poles_pos[:,n] = tract_poles_pos[:,n][idx_sort]
                mask[n]=True
            else:
                tract_poles[:,n] = 0
                tract_poles_pos[:,n] = 0
                tract_freqs[:,n] = 0
                mask[n]=False

        if return_parameters:
            # only compute each physical parameter value for all audio outputs and return that
            # adding dims for glottis_freqs and substract the first coefficients that is always 1
            outdims = glottis_coeffs.shape[0] -1 + 1 + tract_coeffs.shape[0] - 1 + tract_freqs.shape[0] 
            out = np.zeros((audio_input.shape[0], outdims))
            audiorange = np.arange(0,audio_input.shape[0], 1)
            framerange = np.arange(inner_hoplength/2, nframes*inner_hoplength, inner_hoplength)
            curdim = 0
            for nc in range(1,glottis_coeffs.shape[0]):
                out[:,curdim] = np.interp( audiorange, framerange, glottis_coeffs[nc,:] )
                curdim += 1
            out[:,curdim] = np.interp( audiorange, framerange, glottis_freqs )
            curdim += 1
            for nc in range(1,tract_coeffs.shape[0]):
                out[:,curdim] = np.interp( audiorange, framerange, tract_coeffs[nc,:] )
                curdim += 1
            for nc in range(0,tract_freqs.shape[0]):
                out[:,curdim] = np.interp( audiorange, framerange, tract_freqs[nc,:] )
                curdim += 1

            return out

        # TODO convert these metrics to tenseness
        if True: # TODO test more the limits of these changes
            #Glottal shift
            shift = glottis_shift * glottis_freqs.mean()
            glottis_poles_pos = glottis_poles_pos*np.exp(1j * shift) # TODO CHECK IF *= works. calculate glottal shift from tenseness and vocal effort/force
            #Tilt factor
            glottis_poles_real = glottis_poles_real*tilt_factor
            glottis_poles_real = np.where(glottis_poles_real>=1, 0.999, glottis_poles_real)
            #glottis_poles_real = ... # TODO change tilt calculated from vocal effort/force
            glottis_poles = np.stack((glottis_poles_real, glottis_poles_pos, glottis_poles_pos.conj()), axis=0)
            #glottis_poles = np.concatenate((glottis_poles_real.reshape((-1,1)), glottis_poles_pos.reshape((-1,1)), glottis_poles_pos.reshape((-1,1)).conj()), axis=0)
            glottis_coeffs = np.apply_along_axis(np.poly, 0, glottis_poles).real # TODO why does it return complex numbers?

        # apply F1, F2, F3 shifts
        # TODO fix f0 estimation
        tract_shifts_rad = self.shifts_to_freqs(tract_shifts_per, tract_freqs, glottis_freqs*0)  # TODO dirty hack mult by zero!!!      
        tract_poles_pos[0:3, :] *= np.exp(1j * tract_shifts_rad)
        phase_distort = np.array(self.params['knobs'])
        tract_poles_pos = np.apply_along_axis(lambda x: x*np.exp(1j * phase_distort), 0, tract_poles_pos)
        tract_poles = np.concatenate((tract_poles_pos, tract_poles_pos.conj()), axis=0)
        
        # correct way:
        # tract_coeffs = np.apply_along_axis(np.poly, 1, tract_poles) 
        # wrong way but works:
        for n in range(nframes):
            if mask[n]:
                tract_coeffs[:,n] = np.poly(tract_poles[:,n])
        
        # measure distorsion, some how
        distorsion_glottis = np.sum(np.abs(old_glottis_coeffs-glottis_coeffs))/np.sum(np.abs(old_glottis_coeffs))
        distorsion_tract = np.sum(np.abs(old_tract_coeffs-tract_coeffs))/np.sum(np.abs(old_tract_coeffs))
        self.state['distorsion_glottis'] = float(distorsion_glottis)
        self.state['distorsion_tract'] = float(distorsion_tract)

        # regenerate signal
        # fancy new method with denominator and numerator in the filter
        new_glottis_audio = self.filter_frames(input_frames, old_glottis_coeffs, glottis_coeffs)
        audio_output = self.filter_frames(new_glottis_audio, old_tract_coeffs, tract_coeffs)

        # old method of applying two filters
        # new_glottis = self.filter_frames(excitation_signal, np.ones([1]), glottis_coeffs)
        # new_glottis_frames = librosa.util.frame(new_glottis, frame_length=self.framelength, hop_length=self.hoplength)
        # audio_output = self.filter_frames(new_glottis_frames, np.ones([1]), tract_coeffs)
        
        # try renormalizing overlap gain increase
        audio_output *= (self.hoplength / self.framelength)
        
        return audio_output

    def shifts_to_freqs(self, percent_shifts, frequencies_orig, Fg): # convert 3 freqs slider of percentage to frequencies
        percent_shifts = np.array(percent_shifts)*0.99/100 # conversion to -1,1 but not getting quite there so freqz don't overlap
        frames = frequencies_orig.shape[1]
        nfreqs = percent_shifts.shape[0]
        shifts = np.zeros((nfreqs, frames))
        if percent_shifts[0] < 0 and percent_shifts[2] >= 0:
            for n in range(frames):  # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[0:4, n]
                F1 = np.interp(percent_shifts[0], [-1, 0], [Fg[n], F1o])
                F3 = np.interp(percent_shifts[1], [0, 1], [F3o, F4])
                F2 = np.interp(percent_shifts[2], [-1, 0, 1], [F1, F2o, F3])
                shifts[:, n] = [F1 - F1o, F2 - F2o, F3 - F3o]
        if percent_shifts[0] >= 0 and percent_shifts[2] < 0:
            for n in range(frames):  # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[0:4, n]
                F2 = np.interp(percent_shifts[1], [-1, 0, 1], [F1o, F2o, F3o])
                F1 = np.interp(percent_shifts[0], [0, 1], [F1o, F2])
                F3 = np.interp(percent_shifts[2], [-1, 0], [F2, F3o])
                shifts[:, n] = [F1 - F1o, F2 - F2o, F3 - F3o]
        if percent_shifts[0] < 0 and percent_shifts[2] < 0:
            for n in range(frames):  # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[0:4, n]
                F1 = np.interp(percent_shifts[0], [-1, 0], [Fg[n], F1o])
                F2 = np.interp(percent_shifts[1], [-1, 0, 1], [F1, F2o, F3o])
                F3 = np.interp(percent_shifts[2], [-1, 0], [F2, F3o])
                shifts[:, n] = [F1 - F1o, F2 - F2o, F3 - F3o]
        if percent_shifts[0] >= 0 and percent_shifts[2] >= 0:
            for n in range(frames):  # one conversion per frame
                F1o, F2o, F3o, F4 = frequencies_orig[0:4, n]
                F3 = np.interp(percent_shifts[2], [0, 1], [F3o, F4])
                F2 = np.interp(percent_shifts[1], [-1, 0, 1], [F1o, F2o, F3])
                F1 = np.interp(percent_shifts[0], [0, 1], [F1o, F2])
                shifts[:, n] = [F1 - F1o, F2 - F2o, F3 - F3o]

        return shifts 

    def estimate_coeffs(self, data):
        nframes = data.shape[1]

        vtcoeffs = np.empty((self.ncilinders + 1, nframes))
        glcoeffs = np.empty(( 4, nframes))
        lipcoeffs = np.empty((2, nframes))

        for i in range(nframes):
            frame = data[:, i]
            vtcoeffs[:, i], glcoeffs[:, i], lipcoeffs[:, i] = gfm_iaif(frame, n_vt=self.ncilinders)

        return vtcoeffs, glcoeffs, lipcoeffs
    
    def filter_frames(self, data, b, a, framelength=None, hoplength=None):
        if framelength is None:
            framelength = self.framelength
        if hoplength is None:
            hoplength = self.hoplength

        if data.ndim == 1:
            data = librosa.util.frame(data, frame_length=framelength, hop_length=hoplength)
        nframes = data.shape[1]

        if b.ndim == 1:
            b = np.repeat(np.reshape(b, [1, -1]), nframes, axis=1)
        if a.ndim == 1:
            a = np.repeat(np.reshape(a, [1, -1]), nframes, axis=1)

        # TODO: Esta dimension no se cuanto deberia ser si a y b tienen dims diferentes...
        filter_dim = max(b.shape[0],a.shape[0])
        
        if data.ndim == 1:
            out = np.zeros_like(data)
        else:
            out = np.zeros((nframes-1) * hoplength + framelength)

        window = scipy.signal.get_window("hamming", framelength)
        for i in range(nframes):
            frame = data[:, i]
            framepad = np.pad(frame, ((0, filter_dim)), mode="edge")
            
            idx = np.arange(
                librosa.frames_to_samples(i, hop_length=hoplength),
                librosa.frames_to_samples(i, hop_length=hoplength) + framelength,
            )
            #zi = scipy.signal.lfilter_zi(b[:,i], a[:,i])
            #out[idx] += scipy.signal.lfilter(b[:,i], a[:,i], frame, zi=zi*frame[0]) * scipy.signal.get_window("hamming", framelength)
            out[idx] += scipy.signal.lfilter(b[:,i], a[:,i], framepad)[filter_dim :] * window

        return out




"""
MORE UNUSED CODE
# A la excitacion le hemos sacado primero el vocal tract, y ahora le sacamos la glottis
excitation_frames = librosa.util.frame(excitation_iaif, frame_length=framelength, hop_length=hoplength)
# new glottis
exc_plus_glottis = np.zeros_like(audio_input)
for i in range(nframes):  
    frame = excitation_frames[:, i]
    framepad = np.pad(frame, ((0,ncilinders+1)), mode='edge')
    idx = np.arange(librosa.frames_to_samples(i, hop_length=hoplength), librosa.frames_to_samples(i, hop_length=hoplength)+framelength)
    exc_plus_glottis[idx] += scipy.signal.lfilter([1], new_glcoeffs[i,:], framepad)[ncilinders+1:] * scipy.signal.get_window("hamming", framelength)
# new vocal tract
EG_frames = librosa.util.frame(exc_plus_glottis, frame_length=framelength, hop_length=hoplength)
audio_output = np.zeros_like(audio_input)
for i in range(nframes):  
    frame = EG_frames[:, i]
    framepad = np.pad(frame, ((0,ncilinders+1)), mode='edge')
    idx = np.arange(librosa.frames_to_samples(i, hop_length=hoplength), librosa.frames_to_samples(i, hop_length=hoplength)+framelength)
    audio_output[idx] += scipy.signal.lfilter([1], new_vtcoeffs[i,:], framepad)[ncilinders+1:] * scipy.signal.get_window("hamming", framelength)

# O recomponemos a partir del new vocal tract nada mas...
audio_output = np.zeros_like(audio_input)
for i in range(nframes):  
    frame = glottis_frames[:, i]
    framepad = np.pad(frame, ((0,ncilinders+1)), mode='edge')
    idx = np.arange(librosa.frames_to_samples(i, hop_length=hoplength), librosa.frames_to_samples(i, hop_length=hoplength)+framelength)
    audio_output[idx] += scipy.signal.lfilter([1], new_vtcoeffs[i,:], framepad)[ncilinders+1:] * scipy.signal.get_window("hamming", framelength)

















self.update_devices()
self.default_input_device = self.input_devices[self.input_devices_indices.index(sd.default.device[0])]
self.default_output_device = self.output_devices[self.output_devices_indices.index(sd.default.device[1])]

self.stream = sd.Stream(
    channels=2,
    callback=self.audio_callback,
    blocksize=self.block_frame,
    samplerate=self.config.samplerate,
    dtype="float32")
self.stream.start()


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

def set_devices(self, input_device, output_device):
    sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
    sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]
    print("input device:" + str(sd.default.device[0]) + ":" + str(input_device))
    print("output device:" + str(sd.default.device[1]) + ":" + str(output_device))

   def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status):
        start_time = time.perf_counter()
        print("\nStarting callback")
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
        print('sola_shift: ' + str(int(sola_shift)))

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

        outdata[:] = temp_wav[: - self.crossfade_frame, None].repeat(1, 2).cpu().numpy()
        end_time = time.perf_counter()
        print('infer_time: ' + str(end_time - start_time))
        if flag_vc:
            self.window['infer_time'].update(int((end_time - start_time) * 1000))

            

#
# mover F1 entre F0 y F2 (al bajar, jaw closing)
# mover F2 entre F1 y F3 (al subir lengua adelante)
# mover F3 entre F2 y F4 (al bajar cierra labios)
#
# Subir vocal force (higher glottal formant, higher tilt)
# Tenseness wider/higher glotal formant
# 
# 







        # isolate glottis signal
        #glottis_signal = self.filter_frames(input_frames, tract_coeffs, np.ones([1])) ## TODO volver
        glottis_signal = np.zeros_like(audio_input)
        for i in range(nframes):
            frame = input_frames[:, i]
            framepad = np.pad(frame, ((0, self.ncilinders + 1)), mode="edge")
            
            idx = np.arange(
                librosa.frames_to_samples(i, hop_length=self.hoplength),
                librosa.frames_to_samples(i, hop_length=self.hoplength) + self.framelength,
            )

            glottis_signal[idx] += scipy.signal.lfilter(tract_coeffs[i, :], [1], framepad)[self.ncilinders + 1 :] * scipy.signal.get_window("hamming", self.framelength)
        ###
        glottis_frames = librosa.util.frame(glottis_signal, frame_length=self.framelength, hop_length=self.hoplength)

        # isolate excitation signal
        # TODO volver excitation_signal = self.filter_frames(glottis_frames, glottis_coeffs, np.ones([1]))
        excitation_signal = np.zeros_like(audio_input)
        for i in range(nframes):
            frame = glottis_frames[:, i]
            framepad = np.pad(frame, ((0, 3 + 1)), mode="edge")
            
            idx = np.arange(
                librosa.frames_to_samples(i, hop_length=self.hoplength),
                librosa.frames_to_samples(i, hop_length=self.hoplength) + self.framelength,
            )

            excitation_signal[idx] += scipy.signal.lfilter(glottis_coeffs[i, :], [1], framepad)[3 + 1 :] * scipy.signal.get_window("hamming", self.framelength)
        ###        
        excitation_frames = librosa.util.frame(excitation_signal, frame_length=self.framelength, hop_length=self.hoplength)


        # TODO vovler 
        # excitation_frames = librosa.util.frame(excitation_signal, frame_length=self.framelength, hop_length=self.hoplength)
        # new_glottis = self.filter_frames(excitation_frames, np.ones([1]), old_glottis_coeffs)
        new_glottis_signal = np.zeros_like(audio_input)
        for i in range(nframes):
            frame = excitation_frames[:, i]
            framepad = np.pad(frame, ((0, 3 + 1)), mode="edge")
            
            idx = np.arange(
                librosa.frames_to_samples(i, hop_length=self.hoplength),
                librosa.frames_to_samples(i, hop_length=self.hoplength) + self.framelength,
            )

            new_glottis_signal[idx] += scipy.signal.lfilter([1], glottis_coeffs[i, :], framepad)[3 + 1 :] * scipy.signal.get_window("hamming", self.framelength)
        ###        
        
        new_glottis_frames = librosa.util.frame(new_glottis_signal, frame_length=self.framelength, hop_length=self.hoplength)
        # TODO volver 
        # audio_output = self.filter_frames(new_glottis_frames, np.ones([1]), old_tract_coeffs)
        audio_output = np.zeros_like(audio_input)
        for i in range(nframes):
            frame = new_glottis_frames[:, i]
            framepad = np.pad(frame, ((0, self.ncilinders + 1)), mode="edge")
            
            idx = np.arange(
                librosa.frames_to_samples(i, hop_length=self.hoplength),
                librosa.frames_to_samples(i, hop_length=self.hoplength) + self.framelength,
            )

            audio_output[idx] += scipy.signal.lfilter([1], tract_coeffs[i, :], framepad)[self.ncilinders + 1 :] * scipy.signal.get_window("hamming", self.framelength)
        ###         
        
        
        
        # try renormalizing overlap gain increase
        audio_output *= self.hoplength / self.framelength
        # audio_output = audio_input.copy()
        return audio_output







"""
