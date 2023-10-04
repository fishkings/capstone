from flask import Flask, render_template, request,jsonify
from flask import redirect, url_for,Response
from flask import stream_with_context

import cv2
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import os
import time

from streamer import Streamer

print(os.getcwd())
model = load_model("model.h5") # 경로에 한글 없어야 함  

print(model.summary())

application = Flask(__name__)
streamer = Streamer()


# 전처리
def preprocess(frame_test):
    frame_test = np.array(frame_test) # 배열을 numpy 형태로  
    dim = int(np.sqrt(len(frame_test))) # 정사각형 형태의 2차원 이미지 크기 결정
    frame_test = frame_test[:dim*dim].reshape((dim, dim))
    frame_test = cv2.resize(frame_test, (224, 224)) # 이미지 크기 (224,224)
    frame_test_rgb = cv2.cvtColor(frame_test, cv2.COLOR_GRAY2RGB) # 색상 공간 변환
    frame_test_reshaped = np.expand_dims(frame_test_rgb, axis=0) # 배치 차원 생성
    return frame_test_reshaped


# 카메라 처리 
cum_count = 0
def stream_gen( src ):   
    try :    
        streamer.run( src )   
        counter = 0  # 실행 횟수 
        interval = 100  # 간격 설정 (n번 중 한 번 실행)

        global cum_count
        while True :                        
            # 시간 간격을 두고 모델 예측   
            if counter % interval == 0:
                frame_test = streamer.bytescode()[1]
                frame_test_reshaped = preprocess(frame_test)
                class0 = model.predict(frame_test_reshaped)[0][0]
                class1 = model.predict(frame_test_reshaped)[0][1] # 우선 딴 짓으로 생각
                print(class0, class1)
                # 딴 짓 누적 -> 알림
                if class1 > class0 :
                    cum_count +=1
                    print(cum_count)
                # ****************************************************************
                # 딴 짓이 연속적으로 누적되면 AJAX로 신호 보내고 아니면 다시 초기화
                # ****************************************************************

            frame = streamer.bytescode()[1].tobytes()   # 이거는 필요함 
            
            yield (b'--frame\r\n'  # 멀티파트 응답 형식으로 프레임 데이터를 반환
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # yield는 값을 생성하고 반환하는 동시에 실행 상태 유지 가능하게 하는 함수  
            counter += 1
    except GeneratorExit :
        print( 'disconnected stream' )
        streamer.stop()

# 시간 계산
def calculateTime(time ):
        second = int(np.floor(time/1000))
        minute = int(np.floor(second/60))
        hour = int(np.floor(minute/60))
        second %= 60
        minute %= 60
        return f"{hour}시간 {minute}분 {second}초 "

# 기본
@application.route('/')
def index():
    return render_template('index.html')


# ai_recoder 관련 GET,POST
study_timer = None
playingTime = None
differenceTime = None
@application.route('/ai_recoder', methods=['GET','POST'])
def ai_recoder():
    global study_timer
    global differenceTime
    global  playingTime
    
    if request.method == 'POST':
        data = request.get_json()
        study_timer = data['study_timer'] # 공부 시간 (실제 타이머 시간)
        firstclock = data["firstclock"]
        lastclock = data["lastclock"]
        differenceTime = lastclock - firstclock  # 처음 시작 누른 시점 시간 - 처음 종료 누른 시점 시간
        playingTime =  differenceTime -study_timer # 총 시간 - 공부 시간
        
        print("순 공부시간 : ",calculateTime(study_timer))
        print("딴 짓 시간 : ",calculateTime(playingTime))
        print("전체 공부시간 : ",calculateTime(differenceTime))
    return render_template('ai_recoder.html')

@application.route('/recode')
def recode():
    return render_template('recode.html',
                           study_timer=calculateTime(study_timer),
                           playingTime=calculateTime(playingTime),
                           differenceTime=calculateTime(differenceTime))

# 이렇게도 가능...
# 상태 업데이트
@application.route('/update_stream')
def update():
    state = ""
    if cum_count%5 ==0 : 
        state_act = "딴짓 중"
        state_time = 0
    else : 
        state_act = "공부 중"
        state_time = 1
    return jsonify({'state_act': state_act,'state_time':state_time})


# 실시간 스트리밍
@application.route('/stream')
def video_feed(): 
    src = request.args.get( 'src', default = 0, type = int )
    try :  
        return Response(
                        stream_with_context( stream_gen( src ) ),
                        mimetype='multipart/x-mixed-replace; boundary=frame' 
                        # stream_with_context() 함수는 제너레이터가 생성한 데이터를 받아
                        # 적절한 형식으로 Resoponse에 전달
                        )
    except Exception as e : 
        print('stream error : ',str(e))
        

if __name__ == '__main__':
    application.run(debug=True) #host='0.0.0.0', port=8001  서버배포 


# ****************************************************************
# url/recode 만들기 
# 여러명이 접속했을때 카메라 
# @app.route('/test/method/<id>')   -> 카메라 개개인 배치
# def method_test(id):
# ****************************************************************
