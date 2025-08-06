import pygame

pygame.init()

SPLASH_IMAGE = pygame.image.load('./img/splash.png')

SPLASH_SCREEN = pygame.display.set_mode((SPLASH_IMAGE.get_width(), SPLASH_IMAGE.get_height()))
pygame.display.set_caption('Mashup Player')

SPLASH_SCREEN.fill((0, 0, 0))
SPLASH_SCREEN.blit(SPLASH_IMAGE, (0, 0))
pygame.display.flip()

import tkinter as tk
from tkinter import ttk, filedialog
from pydub import AudioSegment
from pygame import mixer
import random
import io
import threading
import os
import numpy as np
from pydub.utils import get_array_type
from pydub.utils import db_to_float
from pydub.effects import normalize
import audioop
import time
import queue
import copy
import math
import librosa
from pymediainfo import MediaInfo
from datetime import datetime
import json

pygame.quit()

MUSIC_EXTENSIONS = ('.mp3', '.m4a', '.opus', '.ogg', '.wav', '.flac')
SOUND_QUEUE_SIZE = 10
FADE_AMOUNT_DB = 80

# These control the granularity of the entropy display
PARTIAL_ENTROPY_SEGMENT_LENGTH_MS = 10000
PARTIAL_ENTROPY_STEP_MS = 1000

class Timer():
    def __init__(self):
        self.start_time = datetime.now()
        self.paused_time = None
        self.paused = False

    def reset(self):
        self.start_time = datetime.now()

    def pause(self):
        if self.paused:
            return
        self.paused_time = datetime.now()
        self.paused = True

    def resume(self):
        if not self.paused:
            return
        pausetime = datetime.now() - self.paused_time
        self.start_time = self.start_time + pausetime
        self.paused = False

    def get(self):
        if self.paused:
            return self.paused_time - self.start_time
        else:
            return datetime.now() - self.start_time

    def get_ms(self):
        td = self.get()
        return td.total_seconds() * 1000

# Taken directly from pydub and modified
def fade_equal_power(self, to_gain=0, from_gain=0):
    # no fade == the same audio
    if to_gain == 0 and from_gain == 0:
        return self
    duration = len(self)
    from_power = db_to_float(from_gain)
    output = []
    gain_delta = db_to_float(to_gain) - from_power
    to_power = from_power + gain_delta
    # fades longer than 100ms can use coarse fading (one gain step per ms),
    # shorter fades will have audible clicks so they use precise fading
    # (one gain step per sample)
    if duration > 100:
        scale_step = gain_delta / duration

        for i in range(duration):
            t = (i / duration)
            volume_change = from_power * (1-t) + to_power * t
            chunk = self[i]
            chunk = audioop.mul(chunk._data,
                                self.sample_width,
                                volume_change ** 0.5)
            output.append(chunk)
    else:
        start_frame = self.frame_count(ms=0)
        end_frame = self.frame_count(ms=duration)
        fade_frames = end_frame - start_frame
        scale_step = gain_delta / fade_frames

        for i in range(int(fade_frames)):
            t = (i / duration)
            volume_change = from_power * (1-t) + to_power * t
            sample = self.get_frame(int(start_frame + i))
            sample = audioop.mul(sample, self.sample_width, volume_change ** 0.5)

            output.append(sample)
    return self._spawn(data=output)

def format_timestamp_ms(ms):
    ms_mod = ms % 1000
    seconds = int(ms / 1000)
    seconds_mod = seconds % 60
    minutes = int(seconds / 60)
    minutes_mod = minutes % 60
    hours = int(minutes / 60)
    if hours > 0:
        return f'{hours:02d}:{minutes_mod:02d}:{seconds_mod:02d}.{ms_mod:03d}'
    else:
        return f'{minutes_mod:02d}:{seconds_mod:02d}.{ms_mod:03d}'

def normalize_segment(segment, target_dBFS=-20.0):
    change_in_dBFS = target_dBFS - segment.dBFS
    return segment.apply_gain(change_in_dBFS)

def audiosegment_to_librosawav(audiosegment):
    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    return fp_arr, len(samples)

class MovingWindowEntropy():
    def __init__(self, range_, bins=256):
        self.bins = bins
        self.range = range_
        self.hist, _ = np.histogram([], bins=self.bins, range=self.range)

    def push(self, arr):
        new_hist, _ = np.histogram(arr, bins=self.bins, range=self.range)
        self.hist += new_hist

    def pop(self, arr):
        old_hist, _ = np.histogram(arr, bins=self.bins, range=self.range)
        self.hist -= old_hist

    def get_entropy(self):
        prob = self.hist / np.sum(self.hist)
        return -np.sum(prob * np.log2(prob + 1e-9)) # +1e-9 to avoid log(0)

def split_by_onsets(sound, offset_ms=-100, min_segment_length_ms=250):
    # return an array of audio segments
    sound_length_ms = len(sound)
    librosa_samples, num_channels = audiosegment_to_librosawav(sound)
    
    sr = sound.frame_rate * num_channels
    t0 = time.time()
    # Changing the hop_length has a roughly quadratic to cubic effect on runtime.
    tempo, beats = librosa.beat.beat_track(y=librosa_samples, sr=sr, hop_length=2048, units='time')
    t1 = time.time()
    split_points_ms = beats * 1000 - offset_ms

    segments = []
    start = 0

    for vv in split_points_ms:
        v = int(vv)
        if v >= sound_length_ms:
            v = sound_length_ms

        # Skip until we get a long enough segment
        if v - start < min_segment_length_ms:
            continue

        segments.append(sound[start:v])
        start = v

    return segments

def sliding_window_entropies(chunks, segment_length_ms, range_, bins):
    entropy_calculator = MovingWindowEntropy(range_=range_, bins=bins)

    segments = [] # (start_ms, end_ms, entropy)

    oldest_sample_idx = -1
    newest_sample_idx = -1
    current_segment_start_ms = 0
    current_segment_end_ms = 0
    num_chunks = len(chunks)

    while True:
        while current_segment_end_ms - current_segment_start_ms < segment_length_ms:
            newest_sample_idx += 1

            if newest_sample_idx >= num_chunks:
                break

            sample, sample_length_ms = chunks[newest_sample_idx]
            entropy_calculator.push(sample)
            current_segment_end_ms += sample_length_ms

        if current_segment_end_ms - current_segment_start_ms >= segment_length_ms:
            entropy = entropy_calculator.get_entropy()
            segments.append((current_segment_start_ms, current_segment_end_ms, entropy))

        if newest_sample_idx >= num_chunks:
            break

        while current_segment_end_ms - current_segment_start_ms >= segment_length_ms:
            oldest_sample_idx += 1
            assert oldest_sample_idx <= newest_sample_idx

            old_sample, old_sample_length_ms = chunks[oldest_sample_idx]
            entropy_calculator.pop(old_sample)
            current_segment_start_ms += old_sample_length_ms

    return segments

def choose_random_segment_by_entropy_beat_aware(song, min_segment_length_ms, top_k_pct=5):
    bit_depth = song.sample_width * 8
    array_type = get_array_type(bit_depth)

    chunks = split_by_onsets(song, offset_ms=-100, min_segment_length_ms=250)

    chunks_samples = [(np.array(chunk.get_array_of_samples(), dtype=array_type), len(chunk)) for chunk in chunks]

    segments = sliding_window_entropies(chunks_samples, min_segment_length_ms, range_=(-2**(bit_depth-1), 2**(bit_depth-1)-1), bins=256)

    if not segments:
        segments.append((0, len(song), 1.0))

    segments_candidates = sorted(segments, key=lambda x: -x[2])[:int(math.ceil(len(segments)*top_k_pct/100))]

    start_ms, end_ms, entropy = random.choice(segments_candidates)

    return song[start_ms:end_ms], start_ms, end_ms, entropy, segments

def compute_partial_entropies_sliding_window(song, segment_length_ms, step_ms):
    bit_depth = song.sample_width * 8
    array_type = get_array_type(bit_depth)

    chunks = [song[i:i+step_ms] for i in range(0, len(song), step_ms)]

    chunks_samples = [(np.array(chunk.get_array_of_samples(), dtype=array_type), len(chunk)) for chunk in chunks]

    segments = sliding_window_entropies(chunks_samples, segment_length_ms, range_=(-2**(bit_depth-1), 2**(bit_depth-1)-1), bins=256)

    if not segments:
        segments.append((0, len(song), 1.0))

    return segments

def clip_to_sound(clip):
    wav_stream = io.BytesIO()
    clip.export(wav_stream, format="wav")
    wav_stream.seek(0)
    return mixer.Sound(wav_stream)

def get_song_name_from_path(path):
    filename = os.path.basename(path)
    parts = filename.split('.')
    name = '.'.join(parts[:-1])
    return name

class SongFile():
    def __init__(self, path):
        self.path = path
        self.name = get_song_name_from_path(path)
        media_info = MediaInfo.parse(path)
        audio_track = None
        for track in media_info.tracks:
            if track.track_type == 'Audio':
                audio_track = track
                break
        self.duration_ms = audio_track.duration
        try:
            self.bitrate_kbps = int(audio_track.bit_rate / 1000)
        except:
            try:
                self.bitrate_kbps = int(media_info.overall_bit_rate / 1000)
            except:
                stat = os.stat(path)
                self.bitrate_kbps = int(stat.st_size / self.duration_ms * 8)
        self.format = audio_track.format

class SoundClipQueueEntry():
    def __init__(self, sound, custom_data):
        self.sound = sound
        self.custom_data = custom_data

class SoundClipProvider(threading.Thread):
    def __init__(self, sound_clip_queue, song_files, shuffle=True):
        threading.Thread.__init__(self, daemon=True)

        self.crossfade_duration_ms = 3000
        self.clip_duration_ms = 20000
        self.sound_clip_queue = sound_clip_queue
        self.shuffle = shuffle
        self.song_list_mutex = threading.Lock()

        self.previous_clip = None

        self.set_song_list(song_files)

    def set_song_list(self, new_list):
        with self.song_list_mutex:
            self.song_list = new_list.copy()
            self.fill_song_queue()

            self.previous_clip = None

    def fill_song_queue(self):
        self.song_queue = queue.SimpleQueue()
        song_list_to_queue = self.song_list.copy()
        if self.shuffle:
            random.shuffle(song_list_to_queue)

        for song in song_list_to_queue:
            self.song_queue.put(song)

    def set_clip_duration_ms(self, ms):
        self.clip_duration_ms = ms

    def set_crossfade_duration_ms(self, ms):
        self.crossfade_duration_ms = ms

    def is_empty(self):
        return not self.song_list

    def run(self):
        while True:
            time.sleep(0.01)

            if not self.song_list:
                time.sleep(0.1)
                continue

            with self.song_list_mutex:
                try:
                    song_file = self.song_queue.get(block=False)
                    song_list = self.song_list
                except:
                    if self.song_list:
                        self.fill_song_queue()
                    continue

            song_path = song_file.path
            crossfade_duration_ms = self.crossfade_duration_ms
            clip_duration_ms = self.clip_duration_ms + crossfade_duration_ms

            song = AudioSegment.from_file(song_path)
            clip, start_ms, end_ms, entropy, segments = choose_random_segment_by_entropy_beat_aware(song, min_segment_length_ms=int(clip_duration_ms), top_k_pct=10)

            # if the song list changed inbetween then discard this one
            if song_list is not self.song_list:
                continue

            partial_entropies = compute_partial_entropies_sliding_window(song, segment_length_ms=PARTIAL_ENTROPY_SEGMENT_LENGTH_MS, step_ms=PARTIAL_ENTROPY_STEP_MS)
            clip = normalize_segment(clip)

            annotation = {
                'start_ms' : start_ms,
                'end_ms' : end_ms - crossfade_duration_ms,
                'entropy' : entropy,
                'entropy_segments' : segments,
                'partial_entropies' : partial_entropies,
                'song_file' : song_file
            }

            with self.song_list_mutex:
                # if the song list changed inbetween then discard this one
                if song_list is not self.song_list:
                    continue

                if self.previous_clip:
                    previous_end_clip = fade_equal_power(self.previous_clip[len(self.previous_clip) - crossfade_duration_ms:], to_gain=-FADE_AMOUNT_DB)
                    next_start_clip = fade_equal_power(clip[:crossfade_duration_ms].fade_in(crossfade_duration_ms), from_gain=-FADE_AMOUNT_DB)
                    self.sound_clip_queue.put(SoundClipQueueEntry(clip_to_sound(previous_end_clip.overlay(next_start_clip, position=0)), custom_data=annotation))

                if self.previous_clip is None:
                    middle_start = 0
                else:
                    middle_start = crossfade_duration_ms
                middle_end = len(clip) - crossfade_duration_ms
                clip_middle = clip[middle_start:middle_end]
                if self.previous_clip is None:
                    clip_middle = clip_middle.fade_in(crossfade_duration_ms)

                self.sound_clip_queue.put(SoundClipQueueEntry(clip_to_sound(clip_middle), custom_data=annotation))

                self.previous_clip = clip


class SoundQueuePlayer(threading.Thread):
    def __init__(self, sound_clip_queue):
        threading.Thread.__init__(self, daemon=True)

        self.sound_clip_queue = sound_clip_queue

        self.audio_channel = mixer.Channel(0)

        self.audio_channel.set_volume(0.3)

        self.current_custom_data = None
        self.queued_custom_data = None

        self.state = 'playing'

        self.state_mutex = threading.Lock()
        self.audio_channel_mutex = threading.Lock()

    def run(self):
        last_added_to_queue = False
        while True:
            time.sleep(0.01)

            with self.audio_channel_mutex:
                if self.audio_channel.get_queue() is None:
                    if last_added_to_queue:
                        self.current_custom_data = self.queued_custom_data
                        self.queued_custom_data = None
                        last_added_to_queue = False

                    try:
                        entry = self.sound_clip_queue.get(block=False)
                    except:
                        entry = None

                    if entry:
                        clip = entry.sound
                        self.audio_channel.queue(clip)
                        self.queued_custom_data = entry.custom_data
                        last_added_to_queue = True

                with self.state_mutex:
                    if self.state == 'paused':
                        self.audio_channel.pause()

    def clear(self):
        with self.audio_channel_mutex:
            self.audio_channel.stop()
            self.current_custom_data = None
            self.queued_custom_data = None

    def get_current_sound(self):
        with self.audio_channel_mutex:
            return self.audio_channel.get_sound()

    def get_current_sound_custom_data(self):
        return self.current_custom_data

    def is_paused(self):
        return self.state == 'paused'

    def unpause(self):
        with self.state_mutex:
            if self.state != 'playing':
                self.state = 'playing'
                self.audio_channel.unpause()

    def pause(self):
        with self.state_mutex:
            if self.state != 'paused':
                self.state = 'paused'
                self.audio_channel.pause()

    def set_volume(self, value):
        with self.audio_channel_mutex:
            self.audio_channel.set_volume(value)


class MashupPlayer():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Mashup Player")
        self.root.geometry('480x480')
        self.root.minsize(480, 480)

        # Frames

        self.currently_playing_frame = tk.Frame(self.root)
        self.currently_playing_frame.pack(side=tk.TOP, fill="both")

        self.controls_frame_1 = tk.Frame(self.root)
        self.controls_frame_1.pack(side=tk.TOP, fill="both")

        self.controls_frame_2 = tk.Frame(self.root)
        self.controls_frame_2.pack(side=tk.TOP, fill="both")

        self.playlist_frame = tk.LabelFrame(self.root, text='Song pool')
        self.playlist_frame.pack(side=tk.TOP, fill="both", expand=1)

        self.status_frame = tk.Frame(self.root)
        self.status_frame.pack(side=tk.TOP, fill="both")

        # Currently playing frame

        self.song_name_label = tk.Label(self.currently_playing_frame)
        self.song_name_label.pack(side=tk.TOP, fill='x')

        self.played_segment_period_label = tk.Label(self.currently_playing_frame)
        self.played_segment_period_label.pack(side=tk.TOP, fill='x')

        self.played_segment_info_label = tk.Label(self.currently_playing_frame)
        self.played_segment_info_label.pack(side=tk.TOP, fill='x')

        self.played_segment_visualization = tk.Canvas(self.currently_playing_frame, height=30)
        self.played_segment_visualization.pack(side=tk.TOP, fill='x')

        # Controls frame

        self.volume_scale = tk.Scale(self.controls_frame_1, from_=0, to=100, label='Volume', length=140, sliderlength=25, orient=tk.HORIZONTAL, command=lambda x: self.set_volume(x))
        self.volume_scale.pack(side=tk.LEFT, padx=5, pady=5)

        self.clip_duration_s_scale = tk.Scale(self.controls_frame_1, from_=10, to=120, label='Clip duration [s]', length=140, sliderlength=25, orient=tk.HORIZONTAL, command=lambda x: self.set_clip_duration_s(x))
        self.clip_duration_s_scale.pack(side=tk.LEFT, padx=5, pady=5)

        self.crossfade_duration_ms_scale = tk.Scale(self.controls_frame_1, from_=500, to=5000, label='Crossfade duration [ms]', length=140, sliderlength=25, orient=tk.HORIZONTAL, command=lambda x: self.set_crossfade_duration_ms(x))
        self.crossfade_duration_ms_scale.pack(side=tk.LEFT, padx=5, pady=5)

        self.play_folder_button = tk.Button(self.controls_frame_2, text="Add Folder", command=lambda: self.add_folder())
        self.play_folder_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_playlist_button = tk.Button(self.controls_frame_2, text="Clear playlist", command=lambda: self.clear_playlist())
        self.clear_playlist_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.play_pause_button = tk.Button(self.controls_frame_2, text="Pause", command=lambda: self.play_pause())
        self.play_pause_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.reset_player_button = tk.Button(self.controls_frame_2, text="Reset queue", command=lambda: self.reset_player())
        self.reset_player_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_config_button = tk.Button(self.controls_frame_2, text="Save config", command=lambda: self.save_config())
        self.save_config_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.load_config_button = tk.Button(self.controls_frame_2, text="Load config", command=lambda: self.load_config())
        self.load_config_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Playlist frame

        self.playlist = ttk.Treeview(self.playlist_frame, columns=('name', 'duration', 'bitrate', 'format'), show='headings')
        self.playlist.column('name')
        self.playlist.column('duration', width=80, stretch=False)
        self.playlist.column('bitrate', width=80, stretch=False)
        self.playlist.column('format', width=80, stretch=False)
        self.playlist.heading('name', text='Name')
        self.playlist.heading('duration', text='Duration')
        self.playlist.heading('bitrate', text='Bitrate [kbps]')
        self.playlist.heading('format', text='Format')
        self.playlist.pack(side=tk.LEFT, fill='both', expand=1)

        self.playlist_scrollbar = ttk.Scrollbar(self.playlist_frame, orient=tk.VERTICAL, command=self.playlist.yview)
        self.playlist.configure(yscroll=self.playlist_scrollbar.set)
        self.playlist_scrollbar.pack(side=tk.RIGHT, fill='y')

        # Status frame

        self.playlist_summary_label = tk.Label(self.status_frame)
        self.playlist_summary_label.pack(side=tk.TOP, anchor='w')

        self.status_label = tk.Label(self.status_frame)
        self.status_label.pack(side=tk.BOTTOM, anchor='w')

        self.root.after(1000, self.update_currently_played_info)
        self.root.after(1000, self.update_playlist_summary)

        # Logic

        self.sound_clip_queue = queue.Queue(10)

        self.sound_clip_provider = SoundClipProvider(sound_clip_queue=self.sound_clip_queue, song_files=[])

        self.sound_queue_player = SoundQueuePlayer(sound_clip_queue=self.sound_clip_queue)

        self.sound_clip_provider.start()
        self.sound_queue_player.start()

        self.volume_scale.set(30)
        self.clip_duration_s_scale.set(20)
        self.crossfade_duration_ms_scale.set(3000)

        self.last_entropy_graph_params = None

        self.num_songs_in_list = 0
        self.total_song_duration_ms = 0

        self.estimate_clip_timer = None
        self.last_song_id = None
        self.segment_visualization_rects = []

    def set_clip_duration_s(self, s):
        self.sound_clip_provider.set_clip_duration_ms(int(s)*1000)

    def set_crossfade_duration_ms(self, ms):
        self.sound_clip_provider.set_crossfade_duration_ms(int(ms))

    def set_volume(self, value):
        # from 0dB to -60dB
        #self.sound_queue_player.set_volume((int(value) / 100) ** 2) # scales better
        if int(value) == 0:
            linear_volume = 0
        else:
            linear_volume = 10 ** (2 * (int(value) / 100 - 1))
        self.sound_queue_player.set_volume(linear_volume) # scales better
        #a = (int(value) / 100) ** 2
        #b = 10 ** (3 * (int(value) / 100 - 1))
        #print(value, a, b)


    def save_config(self):
        path = filedialog.asksaveasfilename()

        config_dict = {
            'volume' : int(self.volume_scale.get()),
            'clip_duration_s' : int(self.clip_duration_s_scale.get()),
            'crossfade_duration_ms' : int(self.crossfade_duration_ms_scale.get()),
            'playlist' : [f.path for f in self.sound_clip_provider.song_list]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f)

    def load_config(self):
        path = filedialog.askopenfilename()
        if not os.path.exists(path):
            return

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        self.volume_scale.set(config_dict['volume'])
        self.clip_duration_s_scale.set(config_dict['clip_duration_s'])
        self.crossfade_duration_ms_scale.set(config_dict['crossfade_duration_ms'])

        self.clear_playlist()

        self.add_paths_to_playlist(config_dict['playlist'])

    def reset_player(self):
        song_list = self.sound_clip_provider.song_list

        self.sound_clip_provider.set_song_list([])

        while not self.sound_clip_queue.empty():
            try:
                self.sound_clip_queue.get(block=False)
            except queue.Empty:
                pass

        self.sound_queue_player.clear()

        self.sound_clip_provider.set_song_list(song_list)

    def clear_playlist(self):
        self.sound_clip_provider.set_song_list([])

        self.total_song_duration_ms = 0
        self.num_songs_in_list = 0

        while not self.sound_clip_queue.empty():
            try:
                self.sound_clip_queue.get(block=False)
            except queue.Empty:
                pass

        self.segment_visualization_rects = []

        self.sound_queue_player.clear()

        try: # in case the gui is not there anymore
            self.playlist_summary_label.configure(text='')
            self.playlist_summary_label.update()

            # Quadratic complexity?
            for row in self.playlist.get_children():
                self.playlist.delete(row)

            self.song_name_label.configure(text='')
            self.played_segment_period_label.configure(text='')
            self.played_segment_info_label.configure(text='')

            self.played_segment_visualization.delete('all')
        except:
            pass

    def update_playlist_summary(self):
        if self.total_song_duration_ms == 0:
            self.playlist_summary_label.configure(text=f'')
        else:
            total_duration_ms = self.total_song_duration_ms
            est_duration_ms = self.sound_clip_provider.clip_duration_ms * self.num_songs_in_list
            self.playlist_summary_label.configure(text=f'Items: {self.num_songs_in_list} | Total duration: {format_timestamp_ms(total_duration_ms)} | Estimated playtime: {format_timestamp_ms(est_duration_ms)}')
        self.root.after(1000, self.update_playlist_summary)

    def add_paths_to_playlist(self, song_list):
        song_files = self.sound_clip_provider.song_list.copy()
        old_song_files = set(f.path for f in song_files)

        for song in song_list:
            path = os.path.normpath(os.path.abspath(song))

            self.status_label.update()

            if path not in old_song_files:
                self.status_label.configure(text=f'Loading {path}.')
                song_file = SongFile(path)
                song_files.append(song_file)
                self.total_song_duration_ms += song_file.duration_ms
                values = (song_file.name, format_timestamp_ms(song_file.duration_ms), song_file.bitrate_kbps, song_file.format)
                self.playlist.insert('', tk.END, values=values)
            else:
                self.status_label.configure(text=f'{path} already in the playlist.')

        self.num_songs_in_list = len(song_files)

        self.status_label.configure(text='')
        self.status_label.update()

        self.sound_clip_provider.set_song_list(song_files)

    def add_folder(self):
        folder_path = filedialog.askdirectory()
        if not os.path.exists(folder_path):
            return

        song_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder_path) for f in filenames if os.path.splitext(f)[1].lower() in MUSIC_EXTENSIONS]
        self.add_paths_to_playlist(song_list)

    def play_pause(self):
        if self.sound_queue_player.is_paused():
            self.sound_queue_player.unpause()
            self.estimate_clip_timer.resume()
            self.play_pause_button.configure(text='Pause')
        else:
            self.sound_queue_player.pause()
            self.estimate_clip_timer.pause()
            self.play_pause_button.configure(text='Play')

    def update_currently_played_info(self):
        data = self.sound_queue_player.get_current_sound_custom_data()
        if data is not None:
            song_file = data['song_file']
            song_duration_ms = song_file.duration_ms
            start_ms = data['start_ms']
            end_ms = data['end_ms']
            song_name = song_file.name
            entropy = data['entropy']
            entropy_segments = data['partial_entropies'] # We use partial entropies so that the window size is constant

            current_song_id = (song_file, start_ms, end_ms)

            self.played_segment_visualization.update()
            cw = self.played_segment_visualization.winfo_width()
            ch = self.played_segment_visualization.winfo_height()

            current_entropy_graph_params = (cw, ch, song_name, start_ms, end_ms)

            if self.last_entropy_graph_params is None or self.last_entropy_graph_params != current_entropy_graph_params:

                self.segment_visualization_rects = []
                self.played_segment_visualization.delete('all')

                min_entropy = min(entropy_segments, key=lambda x: x[2])[2]
                max_entropy = max(entropy_segments, key=lambda x: x[2])[2]
                min_midpoint = (entropy_segments[0][0] + entropy_segments[0][1]) / 2
                max_midpoint = (entropy_segments[-1][0] + entropy_segments[-1][1]) / 2

                last_rect_end_x = -1
                for s_ms, e_ms, e in entropy_segments:
                    midpoint = (s_ms + e_ms) / 2
                    #norm_entropy = (e - min_entropy) / (max_entropy - min_entropy)
                    norm_entropy = 2 ** e / 2 ** max_entropy
                    end_x = int(cw * (midpoint-min_midpoint) / (max_midpoint-min_midpoint))
                    fill_color = 'lawn green' if midpoint >= start_ms and midpoint <= end_ms else 'gray'
                    rect = self.played_segment_visualization.create_rectangle((last_rect_end_x, int(ch-norm_entropy*ch), end_x, ch), fill=fill_color, width=0)
                    if fill_color == 'lawn green':
                        self.segment_visualization_rects.append((midpoint, rect))
                    last_rect_end_x = end_x

                self.last_entropy_graph_params = current_entropy_graph_params

            if self.last_song_id is None or self.last_song_id != current_song_id:
                self.estimate_clip_timer = Timer()
                self.last_song_id = current_song_id
            elif self.estimate_clip_timer:
                # TODO: maybe optimize this with a queue or smth
                for midpoint, rect in self.segment_visualization_rects:
                    if midpoint - start_ms < self.estimate_clip_timer.get_ms():
                        self.played_segment_visualization.itemconfig(rect, fill='green')

            self.song_name_label.configure(text=song_name)

            bitrate_kbps = song_file.bitrate_kbps
            format_ = song_file.format
            self.played_segment_info_label.configure(text=f'{format_} | {bitrate_kbps}kbps')

            self.played_segment_period_label.configure(text=f'{format_timestamp_ms(start_ms)} - {format_timestamp_ms(end_ms)}')

            if not self.sound_clip_provider.is_empty():
                self.status_label.configure(text='')
                self.status_label.update()
        else:
            self.estimate_clip_timer = None
            if not self.sound_clip_provider.is_empty():
                self.status_label.configure(text='Waiting for a song to be processed...')
                self.status_label.update()

        self.root.after(1000, self.update_currently_played_info)

    def run(self):
        self.root.mainloop()
        self.clear_playlist()


if __name__ == "__main__":
    mixer.init()
    player = MashupPlayer()
    player.run()
    mixer.quit()

