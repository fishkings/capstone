from flask import Flask, render_template, request
from flask import redirect, url_for,Response
from flask import stream_with_context
import stack_data

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
model = load_model("model.h5") # 경로에 한글 없어야 함   **********경로 수정*************

print(model.summary())

application = Flask(__name__)
streamer = Streamer()


def stream_gen( src ):   
  
    try :    
        streamer.run( src )   
        while True :

            # 전처리
            frame_test = streamer.bytescode()[1]
            frame_test = np.array(frame_test)  # 배열을 numpy 형태로  
            dim = int(np.sqrt(len(frame_test))) # 정사각형 형태의 2차원 이미지 크기 결정
            frame_test_reshaped = frame_test[:dim*dim].reshape((dim, dim)) # 필요한 만큼 잘라 재배열
            frame_test_resized = cv2.resize(frame_test_reshaped, (224, 224)) # 이미지 크기 (224,224)
            frame_test_rgb = cv2.cvtColor(frame_test_resized, cv2.COLOR_GRAY2RGB) # 색상 공간 변환
            frame_test_reshaped = np.expand_dims(frame_test_rgb, axis=0) # 배치 차원 생성

            frame = streamer.bytescode()[1].tobytes()   # 이거는 필요함 
            
            print(model.predict(frame_test_reshaped))
            
            yield (b'--frame\r\n'  # 멀티파트 응답 형식으로 프레임 데이터를 반환
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # yield는 값을 생성하고 반환하는 동시에 실행 상태 유지 가능하게 하는 함수              
    except GeneratorExit :
        print( 'disconnected stream' )
        streamer.stop()


@application.route('/')
def index():
    return render_template('index.html')

@application.route('/ai_recoder')
def ai_recoder():
    return render_template('ai_recoder.html')

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
    application.run(debug=True)