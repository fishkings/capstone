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

from streamer import Streamer


# ****************************************************************
# //변수 선언//
model = load_model("model.h5") # 경로에 한글 없어야 함  
print(model.summary())

engine = create_engine(r'sqlite:///database.db',echo=False)
application = Flask(__name__)
streamer = Streamer()

Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

# 데이터 클래스 생성
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


# ****************************************************************
# //함수//
# 이미지 전처리
def preprocess_image(frame_test):
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
                frame_test_reshaped = preprocess_image(frame_test)
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

# timestamp 처리 
def format_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H')  # **추후에는 %H도 없애기**

# timestamp를 시간으로 나타내어 문자 출력
def format_time_string(timestamp):
        second = int(np.floor(timestamp/1000))
        minute = int(np.floor(second/60))
        hour = int(np.floor(minute/60))
        second %= 60
        minute %= 60
        return f"{hour}시간 {minute}분 {second}초 "

# timestamp를 '분' 기준으로 변경
def timestamp_to_minutes(timestamp):
    second = int(np.floor(timestamp/1000))
    minute = int(np.floor(second/60))
    second %= 60
    return float(f"{minute}.{second:02}")

# ****************************************************************


# ****************************************************************
# //route//
# 기본
@application.route('/')
def index():
    return render_template('index.html')

# ai_recoder 관련 GET,POST
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

# 실시간 스트리밍
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


# 시간 리스트들 기록
@application.route('/recode')
def recode():
    return render_template('recode.html',
                           studying_time=format_time_string(studying_time),
                           playing_time=format_time_string(playing_time),
                           total_time=format_time_string(total_time))

        
# 차트 기록
# ********************************
# 1. end_time 을 date로 바꿔주기
# 2. studying/playing/total_time은 timestamp로 저장 -> 계산 편의성을 위해
# 2-1. 최종 불러 올때 분으로 변환시키기
# ********************************
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
    
    session.add(talbe_list)  # session.add_all([])  -> 이렇게도 가능
    session.commit()

    # db groupby하여 날짜별 합산 후 리스트에 저장   
    total_time,studying_time,playing_time,end_time = [],[],[],[]   
    searchs = session.query(
        func.sum(timeTable.total_time).label('total_time_sum'),
        func.sum(timeTable.studying_time).label('studying_time_sum'),
        func.sum(timeTable.playing_time).label('playing_time_sum'),
        timeTable.end_time.label('end_time')
    ).group_by(timeTable.end_time).all()

    for search in searchs:
        total_time.append(search[0])
        studying_time.append(search[1])
        playing_time.append(search[2])
        end_time.append(search[3])
        
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
# time_list 값들 sum 하기
# 여러명이 접속했을때 카메라 
# @app.route('/test/method/<id>')   -> 카메라 개개인 배치
# def method_test(id):  -> 이런 방법 시도해보기
# Readme 수정 / 모델 설계도 추가
# 시작 안누르고 바로 종료하면 에러 and 바로 차트 들어가면 에러 -> try 문 쓰기
# 코드 정리 ( 함수는 동사로 시작하게 (함수는 한가지 기능만 수행하도록)/
# 차트로 출력 할 때 timestamp_to_minutes 함수 적용해서 하기 and 하루 넘어갈 때 계산
# ****************************************************************
