from flask import Flask, render_template, redirect, url_for, session, request, Response
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
import time as time_module
import random
import sqlite3
from fn import * 
import shutil

app = Flask(__name__)
app.secret_key = 'hellohelloasdasd!!@!'
app.config['SESSION_COOKIE_NAME'] = 'spotify-login-session'

# Spotify API 인증 정보
SPOTIPY_CLIENT_ID = 'f89717dbf4ae4dbd89ef92d8e174258f'
SPOTIPY_CLIENT_SECRET = '1c6384dd062749f68464852a3524a5ff'
SPOTIPY_REDIRECT_URI = 'http://localhost:5000/callback'

# SpotifyOAuth 객체
sp_oauth = SpotifyOAuth(
    SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI,
    scope='user-read-playback-state,user-modify-playback-state'
)

# 라벨을 정수형으로 변환하는 딕셔너리 생성
label_map = {0: 'Angry', 1: 'Happy', 2: 'Surprise', 3: 'Sad', 4: 'Neutral'}

# 모델 정의 및 로드
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 5)
model.load_state_dict(torch.load('emotion_recognition_model.pth', map_location=torch.device('mps')))
model.eval()

# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Haar 캐스케이드 파일 경로 설정
haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haarcascade_path)

def get_token():
    token_info = sp_oauth.get_cached_token()
    
    if not token_info:
        auth_url = sp_oauth.get_authorize_url()
        return redirect(auth_url)
    
    return token_info

def init_db():
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS emotions
                 (id INTEGER PRIMARY KEY, name TEXT, emotion TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

def save_emotion_to_db(name, emotion):
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    timestamp = time_module.strftime('%Y-%m-%d %H:%M:%S', time_module.localtime())
    c.execute("INSERT INTO emotions (name, emotion, timestamp) VALUES (?, ?, ?)", (name, emotion, timestamp))
    conn.commit()
    conn.close()
    
def get_past_emotions(name):
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute("SELECT emotion, timestamp FROM emotions WHERE name = ? ORDER BY id DESC LIMIT 20", (name,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_song_by_emotion(emotion):
    token_info = get_token()
    
    if not token_info:
        return redirect(url_for('login'))
    
    sp = spotipy.Spotify(auth=token_info['access_token'])
    devices = sp.devices()
    if not devices['devices']:
        return "No active devices found. Please open Spotify on a device and try again."
    
    device_id = devices['devices'][0]['id']
    
    results = sp.search(q=emotion, type='track', limit=10)
    tracks = results['tracks']['items']
    
    if tracks:
        track = random.choice(tracks)
        track_uri = track['uri']
        try:
            sp.start_playback(device_id=device_id, uris=[track_uri])
            track_name = track['name']
            track_artist = track['artists'][0]['name']
            return track_name, track_artist
        except spotipy.exceptions.SpotifyException as e:
            return f"Failed to start playback: {e}"
    else:
        return "No tracks found for the given keyword."

    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('play'))

@app.route('/play', methods=['GET', 'POST'])
def play():
    if request.method == 'POST' or request.args.get('restart'):
        # 웹캠 열기
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        emotion = None
        knn_clf = None
        user_dir = 'user'
        
        if os.path.exists(user_dir) and os.listdir(user_dir):
            # 사용자 디렉토리에 파일이 존재하면 KNN 모델 학습
            knn_clf = train(train_dir=user_dir, model_save_path='knn_model.clf')
        else:
            # KNN 모델 로드
            if os.path.exists('knn_model.clf'):
                with open('knn_model.clf', 'rb') as f:
                    knn_clf = pickle.load(f)
            else:   
                with open('knn_model_basic.clf', 'rb') as f:
                    knn_clf = pickle.load(f)
    
        same_user_count = 0
        last_user = None
        last_emotin = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            X_face_locations = face_recognition.face_locations(frame)
            # 얼굴 인식 및 사용자 이름 찾기
            face_recognition_results = detect_and_recognize(frame, knn_clf)
            
            for name, (top, right, bottom, left) in face_recognition_results:
                if (left-right) * (top-bottom) > 10000: # 가까이 있는 얼굴만 검출되게
                    print(name, emotion)
                    img_path = 'static/now_face.jpg'
                    face = frame[top:bottom, left:right]
                    face = cv2.resize(face, (224, 224))
                    cv2.imwrite(img_path, face)
                    emotion = recognize_emotion(frame, (top, right, bottom, left))
                        
                    if name == last_user and emotion == last_emotin: # 현재 사용자와 감정이 방금 전과 같으면
                        same_user_count += 1
                        last_user = name
                        emotion = last_emotin
                    else:
                        same_user_count = 0
                    last_user = name
                    last_emotin = emotion
                    
                    if same_user_count >= 5:
                        if name == "unknown":
                            return render_template('play.html', name=name, emotion=emotion)
                        else:
                            user_dir = os.path.join('user', name)
                            save_user(user_dir, name)
                            save_emotion_to_db(name, emotion)
                            
                            result = get_song_by_emotion(emotion)
                            if isinstance(result, tuple):
                                track_name, track_artist = result
                                past_emotions = get_past_emotions(name)
                                return render_template('playing.html', name=name, track_name=track_name, track_artist=track_artist, emotion=emotion, past_emotions=past_emotions)
                            else:
                                return result
            
        # 웹캠 및 창 닫기
        cap.release()
        cv2.destroyAllWindows()

        save_user(user_dir, name)
        save_emotion_to_db(name, emotion)
        
        result = get_song_by_emotion(emotion)
        
        if isinstance(result, tuple):
            track_name, track_artist = result
            past_emotions = get_past_emotions(name)
            return render_template('playing.html', name=name, track_name=track_name, track_artist=track_artist, emotion=emotion, past_emotions=past_emotions)
        else:
            return result
    else:
        return render_template('play.html')

# 웹캠 스트리밍
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_popup', methods=['POST'])
def register_popup():
    user = request.form.get('username')
    action = request.form.get('action')
    emotion = request.form.get('emotion')

    if action == 'yes':
        user_dir = os.path.join('user', user)
        
        # 사용자 폴더 생성
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        # static/emotion.jpg 파일을 사용자 폴더로 복사
        src_image_path = 'static/now_face.jpg'
        dst_image_path = os.path.join(user_dir, f'{user}.jpg')
        shutil.copy(src_image_path, dst_image_path)
        
        # 모델 재학습
        knn_clf = train(train_dir='user', model_save_path='knn_model.clf')
        save_emotion_to_db(user, emotion)
        
        # 감정에 맞는 노래 검색 및 재생
        result = get_song_by_emotion(emotion)
        if isinstance(result, tuple):
            track_name, track_artist = result
            past_emotions = get_past_emotions(user)
            return render_template('playing.html', name=user, track_name=track_name, track_artist=track_artist, emotion=emotion, past_emotions=past_emotions)
        else:
            return result
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    app.run(debug=True)