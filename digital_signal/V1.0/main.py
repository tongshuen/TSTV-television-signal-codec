import numpy as np
import scipy.signal as signal
import av
import sys
import os
from typing import Tuple, Optional

class DigitalTVCodec:
    def __init__(self):
        # 视频参数
        self.video_width = 640  # 提高分辨率
        self.video_height = 480
        self.fps = 30  # 提高帧率
        
        # 音频参数
        self.audio_sample_rate = 48000  # 提高音频采样率
        self.audio_channels = 2
        
        # 默认参数
        self.default_carrier_freq = 438.5  # 默认载波频率 (MHz)
        self.default_sample_rate = 6  # 提高默认采样率 (MS/s)
        
        # 数字调制参数
        self.symbol_rate = 2e6  # 提高符号率
        self.qam_order = 64  # 升级到64-QAM以提高数据率
        self.color_depth = 6  # 每个颜色通道6位 (RGB共18位)
    
    def parse_frequency(self, freq_str: str) -> float:
        """解析频率字符串，必须包含单位"""
        freq_str = freq_str.lower().strip()
        if freq_str.endswith('mhz'):
            return float(freq_str[:-3])
        elif freq_str.endswith('khz'):
            return float(freq_str[:-3]) / 1000
        elif freq_str.endswith('hz'):
            return float(freq_str[:-2]) / 1e6
        else:
            raise ValueError("频率必须显式指定单位 (MHz, kHz 或 Hz)")
    
    def parse_sample_rate(self, rate_str: str) -> float:
        """解析采样率字符串，必须包含单位"""
        rate_str = rate_str.lower().strip()
        if rate_str.endswith('ms/s') or rate_str.endswith('msps'):
            return float(rate_str[:-4])
        elif rate_str.endswith('ks/s') or rate_str.endswith('ksps'):
            return float(rate_str[:-4]) / 1000
        elif rate_str.endswith('s/s') or rate_str.endswith('sps'):
            return float(rate_str[:-3]) / 1e6
        else:
            raise ValueError("采样率必须显式指定单位 (MS/s, kS/s 或 S/s)")
    
    def determine_io_files(self) -> Tuple[str, str]:
        """确定输入输出文件"""
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
            if len(sys.argv) > 2:
                output_file = sys.argv[2]
            else:
                output_file = 'output.mkv' if input_file.endswith('.iq') else 'output.iq'
        else:
            if os.path.exists('input.iq'):
                input_file = 'input.iq'
                output_file = 'output.mkv'
            elif os.path.exists('input.mkv'):
                input_file = 'input.mkv'
                output_file = 'output.iq'
            else:
                raise FileNotFoundError("没有找到默认输入文件 (input.iq 或 input.mkv)")
        
        return input_file, output_file
    
    def get_parameters(self) -> Tuple[float, float]:
        """获取载波频率和采样率参数"""
        if len(sys.argv) > 3:
            carrier_freq = self.parse_frequency(sys.argv[3])
        else:
            carrier_freq = self.default_carrier_freq
        
        if len(sys.argv) > 4:
            sample_rate = self.parse_sample_rate(sys.argv[4])
        else:
            sample_rate = self.default_sample_rate
        
        return carrier_freq, sample_rate
    
    def quantize_color(self, rgb_frame: np.ndarray) -> np.ndarray:
        """量化RGB颜色到指定位深"""
        max_val = (1 << self.color_depth) - 1
        quantized = np.round(rgb_frame / 255 * max_val).astype(np.uint8)
        return quantized
    
    def dequantize_color(self, quantized: np.ndarray) -> np.ndarray:
        """反量化颜色数据"""
        max_val = (1 << self.color_depth) - 1
        rgb_frame = (quantized / max_val * 255).astype(np.uint8)
        return rgb_frame
    
    def qam_modulate(self, data: np.ndarray) -> np.ndarray:
        """64-QAM调制"""
        # 将数据分成6位一组 (64-QAM)
        data_6bit = np.zeros(len(data) // 6 * 6, dtype=np.uint8)
        for i in range(6):
            data_6bit |= (data[i::6] & 0x01) << i
        
        # 64-QAM星座图
        constellation = np.array([
            -7-7j, -7-5j, -7-3j, -7-1j, -7+7j, -7+5j, -7+3j, -7+1j,
            -5-7j, -5-5j, -5-3j, -5-1j, -5+7j, -5+5j, -5+3j, -5+1j,
            -3-7j, -3-5j, -3-3j, -3-1j, -3+7j, -3+5j, -3+3j, -3+1j,
            -1-7j, -1-5j, -1-3j, -1-1j, -1+7j, -1+5j, -1+3j, -1+1j,
            7-7j,  7-5j,  7-3j,  7-1j,  7+7j,  7+5j,  7+3j,  7+1j,
            5-7j,  5-5j,  5-3j,  5-1j,  5+7j,  5+5j,  5+3j,  5+1j,
            3-7j,  3-5j,  3-3j,  3-1j,  3+7j,  3+5j,  3+3j,  3+1j,
            1-7j,  1-5j,  1-3j,  1-1j,  1+7j,  1+5j,  1+3j,  1+1j
        ]) / np.sqrt(42)  # 归一化功率
        
        return constellation[data_6bit]
    
    def qam_demodulate(self, signal: np.ndarray) -> np.ndarray:
        """64-QAM解调"""
        constellation = np.array([
            -7-7j, -7-5j, -7-3j, -7-1j, -7+7j, -7+5j, -7+3j, -7+1j,
            -5-7j, -5-5j, -5-3j, -5-1j, -5+7j, -5+5j, -5+3j, -5+1j,
            -3-7j, -3-5j, -3-3j, -3-1j, -3+7j, -3+5j, -3+3j, -3+1j,
            -1-7j, -1-5j, -1-3j, -1-1j, -1+7j, -1+5j, -1+3j, -1+1j,
            7-7j,  7-5j,  7-3j,  7-1j,  7+7j,  7+5j,  7+3j,  7+1j,
            5-7j,  5-5j,  5-3j,  5-1j,  5+7j,  5+5j,  5+3j,  5+1j,
            3-7j,  3-5j,  3-3j,  3-1j,  3+7j,  3+5j,  3+3j,  3+1j,
            1-7j,  1-5j,  1-3j,  1-1j,  1+7j,  1+5j,  1+3j,  1+1j
        ]) / np.sqrt(42)
        
        distances = np.abs(signal[:, np.newaxis] - constellation)
        symbols = np.argmin(distances, axis=1)
        
        # 将符号转换回6位数据
        data = np.zeros(len(symbols) * 6, dtype=np.uint8)
        for i in range(6):
            data[i::6] = (symbols >> i) & 0x01
        
        return data
    
    def pack_rgb_data(self, rgb_quantized: np.ndarray) -> np.ndarray:
        """打包RGB量化数据为比特流"""
        # 每个像素18位 (6位/通道 × 3通道)
        bits = np.unpackbits(rgb_quantized.reshape(-1, 1), axis=1)[:, -6:]
        return bits.flatten()
    
    def unpack_rgb_data(self, bitstream: np.ndarray, frame_size: Tuple[int, int]) -> np.ndarray:
        """从比特流解包RGB数据"""
        h, w = frame_size
        bits = bitstream.reshape(-1, 6)
        rgb_quantized = np.packbits(np.pad(bits, ((0, 0), (2, 0)), 'constant'), axis=1)
        return rgb_quantized.reshape(h, w, 3)
    
    def mkv_to_iq(self, mkv_path: str, carrier_freq: float, sample_rate: float) -> np.ndarray:
        """将MKV文件编码为数字IQ信号"""
        print(f"编码MKV到数字IQ: {mkv_path}")
        print(f"载波频率: {carrier_freq} MHz")
        print(f"采样率: {sample_rate} MS/s")
        
        with av.open(mkv_path) as container:
            # 获取视频流
            video_stream = next(s for s in container.streams if s.type == 'video')
            video_frames = []
            
            # 读取并量化视频帧
            for frame in container.decode(video=0):
                if frame.width != self.video_width or frame.height != self.video_height:
                    frame = frame.reformat(self.video_width, self.video_height)
                rgb_frame = frame.to_ndarray(format='rgb24')
                quantized = self.quantize_color(rgb_frame)
                video_frames.append(quantized)
            
            # 获取音频流
            audio_data = None
            try:
                audio_stream = next(s for s in container.streams if s.type == 'audio')
                audio_frames = [frame.to_ndarray() for frame in container.decode(audio=0)]
                if audio_frames:
                    audio_data = np.concatenate(audio_frames)
                    if audio_data.ndim == 1:
                        audio_data = np.column_stack((audio_data, audio_data))
            except StopIteration:
                pass
            
            # 如果没有音频，创建静音
            if audio_data is None:
                duration = len(video_frames) / self.fps
                num_samples = int(duration * self.audio_sample_rate)
                audio_data = np.zeros((num_samples, 2), dtype=np.float32)
        
        # 打包视频数据为比特流
        video_bitstream = np.concatenate([self.pack_rgb_data(frame) for frame in video_frames])
        
        # 处理音频数据
        audio_mono = audio_data.mean(axis=1)
        # 量化音频到6位 (0-63)
        audio_quantized = np.round((audio_mono + 1) * 31.5).astype(np.uint8)
        
        # 确保数据长度匹配
        min_length = min(len(video_bitstream), len(audio_quantized) * 3)  # 音频数据量较少
        video_data = video_bitstream[:min_length]
        audio_data = np.repeat(audio_quantized[:min_length // 3], 3)
        
        # 交错视频和音频数据
        combined_data = np.empty(len(video_data) + len(audio_data), dtype=np.uint8)
        combined_data[0::2] = video_data  # 偶数索引放视频
        combined_data[1::2] = audio_data  # 奇数索引放音频
        
        # 数字调制
        modulated = self.qam_modulate(combined_data)
        
        # 上变频到载波频率
        t = np.arange(len(modulated)) / (sample_rate * 1e6)
        carrier = np.exp(2j * np.pi * carrier_freq * 1e6 * t)
        rf_signal = modulated * carrier
        
        # 分离I和Q
        iq_data = np.column_stack((rf_signal.real, rf_signal.imag))
        
        return iq_data
    
    def iq_to_mkv(self, iq_path: str, carrier_freq: float, sample_rate: float) -> None:
        """将数字IQ信号解码为MKV文件"""
        print(f"解码数字IQ到MKV: {iq_path}")
        print(f"载波频率: {carrier_freq} MHz")
        print(f"采样率: {sample_rate} MS/s")
        
        # 加载IQ数据
        iq_data = np.fromfile(iq_path, dtype=np.float32).reshape(-1, 2)
        rf_signal = iq_data[:, 0] + 1j * iq_data[:, 1]
        
        # 下变频
        t = np.arange(len(rf_signal)) / (sample_rate * 1e6)
        carrier = np.exp(-2j * np.pi * carrier_freq * 1e6 * t)
        baseband = rf_signal * carrier
        
        # 数字解调
        bitstream = self.qam_demodulate(baseband)
        
        # 分离视频和音频数据
        video_bitstream = bitstream[0::2]  # 偶数索引是视频
        audio_bits = bitstream[1::2]      # 奇数索引是音频
        
        # 视频处理
        pixels_per_frame = self.video_width * self.video_height
        bits_per_frame = pixels_per_frame * 18  # 每个像素18位
        num_frames = len(video_bitstream) // bits_per_frame
        video_bitstream = video_bitstream[:num_frames * bits_per_frame]
        
        # 解包视频帧
        video_frames = []
        for i in range(num_frames):
            frame_bits = video_bitstream[i*bits_per_frame : (i+1)*bits_per_frame]
            quantized = self.unpack_rgb_data(frame_bits, (self.video_height, self.video_width))
            rgb_frame = self.dequantize_color(quantized)
            video_frames.append(rgb_frame)
        
        # 音频处理
        # 合并音频位 (每6位一个样本)
        audio_samples = audio_bits[:len(audio_bits) // 6 * 6].reshape(-1, 6)
        audio_quantized = np.packbits(np.pad(audio_samples, ((0, 0), (2, 0)), 'constant'), axis=1)
        # 反量化到-1到1
        audio_samples = (audio_quantized.flatten() / 63.0 * 2 - 1).astype(np.float32)
        # 重采样到目标采样率
        audio_samples = signal.resample(
            audio_samples,
            int(len(audio_samples) * self.audio_sample_rate / (sample_rate * 1e6 * 0.5))
        )
        # 裁剪到16位
        audio_samples = np.clip(audio_samples, -1.0, 1.0)
        audio_samples = (audio_samples * 32767).astype(np.int16)
        # 创建双声道
        audio_data = np.column_stack((audio_samples, audio_samples))
        
        # 创建MKV文件
        output_path = iq_path.replace('.iq', '.mkv')
        with av.open(output_path, 'w') as container:
            # 添加视频流 (FFV1编码)
            video_stream = container.add_stream('ffv1', rate=self.fps)
            video_stream.width = self.video_width
            video_stream.height = self.video_height
            video_stream.pix_fmt = 'rgb24'
            
            # 添加音频流 (FLAC编码)
            audio_stream = container.add_stream('flac', rate=self.audio_sample_rate)
            audio_stream.channels = self.audio_channels
            
            # 写入视频帧
            for frame in video_frames:
                av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
                for packet in video_stream.encode(av_frame):
                    container.mux(packet)
            
            # 结束视频流
            for packet in video_stream.encode():
                container.mux(packet)
            
            # 写入音频
            audio_frame = av.AudioFrame.from_ndarray(
                audio_data.T,
                format='s16',
                layout='stereo'
            )
            audio_frame.rate = self.audio_sample_rate
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)
            
            # 结束音频流
            for packet in audio_stream.encode():
                container.mux(packet)
        
        print(f"MKV文件已保存: {output_path}")
    
    def run(self):
        """运行编码器/解码器"""
        try:
            input_file, output_file = self.determine_io_files()
            carrier_freq, sample_rate = self.get_parameters()
            
            if input_file.endswith('.iq'):
                self.iq_to_mkv(input_file, carrier_freq, sample_rate)
            elif input_file.endswith('.mkv'):
                iq_data = self.mkv_to_iq(input_file, carrier_freq, sample_rate)
                iq_data.tofile(output_file)
                print(f"数字IQ文件已保存: {output_file}")
            else:
                raise ValueError("输入文件必须是 .iq 或 .mkv 格式")
        except Exception as e:
            print(f"错误: {str(e)}")
            print("用法: python dtv_codec.py [input] [output] [carrier_freq] [sample_rate]")
            print("示例: python dtv_codec.py input.mkv output.iq 438.5MHz 6MS/s")
            sys.exit(1)

if __name__ == "__main__":
    codec = DigitalTVCodec()
    codec.run()
