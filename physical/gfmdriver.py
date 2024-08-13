from .gfm import Resynth
import json
class GFMDriver:
    model = None

    def __init__(self,samplerate=None, framelength=None, hoplength=None):
        self.model = Resynth()
        self.refresh_devices()
        self.inputs, self.outputs = self.model.get_devices()
        self.cur_input = self.inputs[1]
        self.cur_output = self.outputs[0]
        self.model.set_devices(self.cur_input,self.cur_output)
        self.audio = None
        self.samplerate = samplerate
        self.framelength = framelength
        self.hoplength = hoplength
        self.nknobs = self.model.nknobs

    def refresh_devices(self):
        self.model.update_devices()
        self.inputs, self.outputs = self.model.get_devices()
        for inp in self.inputs:
            if inp.find("Air")>=0:
                self.cur_input = inp
                break
            else:
                self.cur_input = self.inputs[1]
        for out in self.outputs:
            if out.find("Headphones")>=0:
                self.cur_output = out
                break
            else:
                self.cur_output = self.outputs[0]

    def update_selected_input(self, sel_input):
        self.cur_input = sel_input.value
        self.model.set_devices(self.cur_input,self.cur_output)

    def update_selected_output(self, sel_output):
        self.cur_output = sel_output.value
        self.model.set_devices(self.cur_input, self.cur_output)  

    def update_value(self,key, value):
        if key == "F1":
            self.model.params['vt_shifts'][0]=value
        elif key == "F2":
            self.model.params['vt_shifts'][1]=value
        elif key == "F3":
            self.model.params['vt_shifts'][2]=value
        elif key == "F0":
            self.model.params['glottis_shifts']=value
        elif key == "tilt":
            self.model.params['tilt_factor']=value

    def get_value(self,key):
        if key == "F1":
            return self.model.params['vt_shifts'][0]
        elif key == "F2":
            return self.model.params['vt_shifts'][1]
        elif key == "F3":
            return self.model.params['vt_shifts'][2]
        elif key == "F0":
            return self.model.params['glottis_shifts']
        elif key == "tilt":
            return self.model.params['tilt_factor']

    def distorsion_measure(self):
        return self.model.distorsion

    def update_knob(self, n, value):
        self.model.params['knobs'][n]=value

    def save_params(self):
        return json.dumps(self.model.params).encode("utf-8")


    def load_params(self,filename):
        pass

    def get_latency(self):        
        return self.model.get_latency()
    
    def start_process(self):
        self.model.start_stream()

    def stop_process(self):
        self.model.stop_stream()
    
    def store_audio(self, data, samplerate):
        self.audio = data
        self.samplerate = samplerate
        print("STORED",type(self.audio),self.audio.shape,"at",self.samplerate)

    def play_audio(self,actually_play=True):                
        self.model.play_audio(self.audio,self.samplerate,actually_play)

    def stream_audio(self):                
        self.model.play_audio(self.audio,self.samplerate)        

    def set_framelen(self,val):
        self.framelength = val.value
        self.model.framelength=val.value

    def set_samplerate(self,val):
        self.samplerate = val

    def set_hoplen(self,val):
        self.model.hoplength=val.value

    def set_blocks(self,val):
        self.model.process_blocks=int(val.value)

    def extract(self, audio, pad=False):
        self.model.fs = self.samplerate
        self.model.framelength = self.framelength
        self.model.hoplength = self.hoplength
        return self.model.process( audio, return_parameters=True, pad=pad)

### 