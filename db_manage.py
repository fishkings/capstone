
import sqlalchemy
from sqlalchemy import create_engine, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Integer,String
from sqlalchemy.orm import sessionmaker
import uuid


engine = create_engine(r'sqlite:///database.db',echo=False)

Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()


class timeTable(Base):
    __tablename__ = 'studyTable'
    id = Column(String, primary_key =True, default=uuid.uuid4().hex)  #고유성 제약 조건 (동일한 키 안됨)
    start_time = Column(String)
    end_time = Column(String)  
    date = Column(String)
    playing_time = Column(Integer)
    studying_time = Column(Integer)
    total_time = Column(Integer)
timeTable.__table__.create(bind=engine, checkfirst=True)

# 기존 테이블 삭제 / 생성
# Base.metadata.drop_all(engine)
# Base.metadata.create_all(engine)

# READ
result = session.query(timeTable).all()
for row in result:
    print(f"\n시작시간: {row.start_time} | 종료 시간: {row.end_time} \
    \n 총 시간 : {row.total_time} | 공부 시간 : {row.studying_time} | 딴짓 시간 : {row.playing_time}")

    
studying_time = [x for x in session.query(timeTable.studying_time).all()]
playing_time = [x for x in session.query(timeTable.playing_time).all()]
total_time = [x for x in session.query(timeTable.total_time).all()]
date = [x for x in session.query(timeTable.end_time).all()]
origin_datasets = {'data' : [studying_time,playing_time,total_time,date]} 

playing_time = [origin_dataset[0] for origin_dataset in origin_datasets['data'][0]]
studying_time = [origin_dataset[0] for origin_dataset in origin_datasets['data'][1]]
total_time = [origin_dataset[0] for origin_dataset in origin_datasets['data'][2]]
date = [origin_dataset[0] for origin_dataset in origin_datasets['data'][3]]

datasets = [
        {'label' : date , # 임시로 end_time 설정
         'playing_time' : playing_time,
         'studying_time' : studying_time,
         'total_time' : total_time}
    ]    
print("\n\n",datasets)