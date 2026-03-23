### 🎵 Facial Emotion Detect 노래 추천 웹사이트

**문제 정의**  
사용자의 표정에서 감정을 예측하고, 사용자 이력까지 반영해 감정 기반 음악을 추천하는 웹사이트를 구현했습니다.

**핵심 기능**
- ResNet18 기반 얼굴 표정 인식 및 감정 예측
- K-NN 기반 사용자 식별
- Spotify API 기반 음악 추천 및 재생
- Flask 기반 웹페이지 구현
- SQLite3 기반 이용 기록 저장

**사용 기술**  
![ResNet18](https://img.shields.io/badge/ResNet18-1F6FEB?style=flat-square)
![KNN](https://img.shields.io/badge/K--NN-8250DF?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![SQLite3](https://img.shields.io/badge/SQLite3-003B57?style=flat-square&logo=sqlite&logoColor=white)
![Spotify](https://img.shields.io/badge/Spotify%20API-1DB954?style=flat-square&logo=spotify&logoColor=white)

**기여도**
- 감정 예측 모델 학습: 10/10
- 사용자 인식 모델 및 데이터베이스 구축: 10/10
- 웹페이지 구현: 5/10

**문제 해결**
1. 초기 표정 인식 모델의 정확도가 낮았습니다.  
   → 기존 데이터셋이 서양인 및 흑백 위주라는 점을 분석했습니다.  
   → 동양인 비중이 높은 데이터셋을 추가하고, angry / neutral 클래스 수 불균형을 확인했습니다.  
   → 가중치 조정과 data augmentation(cut-out, crop, noise 등)을 적용해 모든 클래스에서 85% 이상의 정확도를 달성했습니다.
2. 사용자 식별에 ResNet 계열 모델을 사용하면 재학습 시간이 길고 데이터도 충분하지 않았습니다.  
   → K-NN을 활용해 빠르게 사용자 데이터를 구축하는 방식으로 전환했습니다.
