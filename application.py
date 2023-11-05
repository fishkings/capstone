from flask import Flask, render_template, request,jsonify
from flask import redirect, url_for,Response
from flask import stream_with_context

import sqlalchemy
from sqlalchemy import func
from sqlalchemy import create_engine, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Integer,String
from sqlalchemy.orm import sessionmaker
import uuid

import cv2
import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

import os
import time
from time import strftime
from time import localtime
from datetime import datetime
from joblib import dump, load
import warnings


from streamer import Streamer

warnings.filterwarnings("ignore")

# [변수 선언]
model = load('model.joblib') # 경로에 한글 없어야 함  
# movenet 모델 interpreter 
interpreter = tf.lite.Interpreter(model_path="movenet/lite-model_movenet_singlepose_lightning_3.tflite")
interpreter.allocate_tensors()

engine = create_engine(r'sqlite:///database.db',echo=False)
application = Flask(__name__)
streamer = Streamer()

Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

# //데이터 클래스 생성
class timeTable(Base):
    __tablename__ = 'studyTable'
    id = Column(String, primary_key =True, default=uuid.uuid4().hex)  #고유성 제약 조건 (동일한 키 안됨)
    initial_time = Column(String)
    end_time = Column(String)  
    date = Column(String)
    playing_time = Column(Integer)
    studying_time = Column(Integer)
    total_time = Column(Integer)
timeTable.__table__.create(bind=engine, checkfirst=True)



# [함수]
# //카메라 처리 
state = 0
interval = 0  # 모델 predict 딜레이

def stream_gen( src ):    
    try :    
        streamer.run( src )   
        counter = 0  # 실행 횟수 
        interval = 100  # 간격 설정 (n번 중 한 번 실행)

        global state
        while True :   
            data = []                     
            frame,image = streamer.bytescode()

            if interval % 40 == 0:
                # Reshape image
                image = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 192,192)
                input_image = tf.cast(image, dtype=tf.float32)

                # Setup input and output 
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Make predictions 
                interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
                interpreter.invoke()
                keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
            
                point_cnt = 0  # 관절수 카운트
                for keypoints_with_score in keypoints_with_scores[0][0][:11]:
                    for ax in range(2):
                        data.append(keypoints_with_score[:2][ax])
                        if (keypoints_with_score[2]> 0.1):# 0.1 이하는 point 감지 못 한 것으로 간주
                            point_cnt += 1

                # 모델 테스트
                data = np.array(data)
                state = model.predict(data.reshape(1, -1))
                print(f"감지되는 관절 포인트: {point_cnt}")
                if point_cnt >= 16 :   #  2개(x,y좌표) * 3(최소 3관절 이상) * 2(오른쪽,왼쪽)
                    if state == 1 : 
                        print("공부 중")
                    elif state == 0 : 
                        print("딴짓 중")
                else : 
                    state = 0 
                    print("딴짓 중 _ 관절수 부족")
                    
                interval = 0 
            interval += 1 

            
            yield (b'--frame\r\n'  # 멀티파트 응답 형식으로 프레임 데이터를 반환
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # yield는 값을 생성하고 반환하는 동시에 실행 상태 유지 가능하게 하는 함수   

    except GeneratorExit :
        print( 'disconnected stream' )
        streamer.stop()


# //timestamp 처리 
def format_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H')  # **추후에는 %H도 없애기**


# //timestamp를 시간으로 나타내어 문자 출력
def format_time_string(timestamp):
        second = int(np.floor(timestamp/1000))
        minute = int(np.floor(second/60))
        hour = int(np.floor(minute/60))
        second %= 60
        minute %= 60
        return f"{hour}시간 {minute}분 {second}초 "


# //timestamp를 '분' 기준으로 변경
def timestamp_to_minutes(timestamp):
    second = int(np.floor(timestamp/1000))
    minute = int(np.floor(second/60))
    second %= 60
    return float(f"{minute}.{second:02}")



# [route]
# // 기본 index
@application.route('/')
def index():
    return render_template('index.html')


# //ai_recoder 관련 GET,POST
studying_time, playing_time, total_time, initial_timestamp, end_timestamp = None, None, None, None, None

@application.route('/ai_recoder', methods=['GET','POST'])
def ai_recoder():
    global studying_time
    global total_time
    global playing_time
    global initial_timestamp  
    global end_timestamp
    
    if request.method == 'POST':
        data = request.get_json()
        studying_time = data['studying_time'] # 공부 시간 (실제 타이머 시간)
        initial_timestamp = data["initial_timestamp"]
        end_timestamp = data["end_timestamp"]
        total_time = end_timestamp - initial_timestamp  # 처음 시작 누른 시점 시간 - 처음 종료 누른 시점 시간
        playing_time =  total_time -studying_time # 총 시간 - 공부 시간
        
        print(studying_time,end_timestamp, initial_timestamp)
        print("순 공부시간 : ",format_time_string(studying_time))
        print("딴 짓 시간 : ",format_time_string(playing_time))
        print("전체 공부시간 : ",format_time_string(total_time))
    return render_template('ai_recoder.html')


# //실시간 스트리밍
@application.route('/stream')
def stream(): 
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


# //상태 업데이트
@application.route('/update_stream')
def update():
    if state : 
        state_act = "공부 중"
        state_time = 1        
    else : 
        state_act = "딴짓 중"
        state_time = 0
    return jsonify({'state_act': state_act,'state_time':state_time}) # 이렇게도 가능...


# //시간 리스트들 기록
@application.route('/recode')
def recode():
    return render_template('recode.html',
                           studying_time=format_time_string(studying_time),
                           playing_time=format_time_string(playing_time),
                           total_time=format_time_string(total_time))

        
# //차트 기록
@application.route('/recode_chart',methods=['GET','POST'])
def recode_chart():
    global studying_time 
    global total_time
    global playing_time
    global end_time  # 수정되야 할 부분

    initial_time = format_timestamp(initial_timestamp/1000)
    end_time = format_timestamp(end_timestamp/1000)
    # date = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d') 
   
    talbe_list = timeTable(id=str(uuid.uuid4().hex),
                           # **date 값 추가 해야함**
                            initial_time=initial_time,  
                            end_time=end_time,
                            studying_time=studying_time,
                            playing_time=playing_time,
                            total_time=total_time)
    
    session.add(talbe_list)  
    session.commit()

    # db groupby하여 날짜별 합산    
    total_time,studying_time,playing_time,end_time = [],[],[],[]   
    time_list_sums = session.query(
        func.sum(timeTable.total_time).label('total_time_sum'),
        func.sum(timeTable.studying_time).label('studying_time_sum'),
        func.sum(timeTable.playing_time).label('playing_time_sum'),
        timeTable.end_time.label('end_time')
    ).group_by(timeTable.end_time).all()

    # 리스트에 저장 후 timestamp를 minutes으로 변경
    for time_list_sum in time_list_sums:
        total_time.append(timestamp_to_minutes(time_list_sum[0]))
        studying_time.append(timestamp_to_minutes(time_list_sum[1]))
        playing_time.append(timestamp_to_minutes(time_list_sum[2]))
        end_time.append(time_list_sum[3])
        
    print(total_time,studying_time,playing_time,end_time)

    # gruopby 안하고 매번 카운트 하는 경우
    # time_lists = ["playing_time", "studying_time",  "total_time", "end_time"] 
    # for idx,time_list in enumerate(time_lists):
    #     datas = [x for x in session.query(getattr(timeTable, time_list)).all()] # 동적 할당을 위해 getattr 사용
    #     globals()[time_list] =  [data[0] for data in datas] # playing/studying/total_time에 전처리 후 데이터 할당

    datasets = [
        {'label' : end_time , # 수정되야 할 부분
         'playing_time' : playing_time,
         'studying_time' : studying_time,
         'total_time' : total_time}
    ]    
    return render_template('recode_chart.html',
                              datasets = datasets)



if __name__ == '__main__':
    application.run(debug=True) #host='0.0.0.0', port=8001  서버배포 


# ****************************************************************
# 여러명이 접속했을때 카메라 
# @app.route('/test/method/<id>')   -> 카메라 개개인 배치
# def method_test(id):  -> 이런 방법 시도해보기
# Readme 수정 / 모델 설계도 추가
# 시작 안누르고 바로 종료하면 에러 and 바로 차트 들어가면 에러 -> try 문 쓰기
# 코드 정리 ( 함수는 동사로 시작하게 (함수는 한가지 기능만 수행하도록)/
# ****************************************************************
