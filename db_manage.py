
import sqlalchemy
from sqlalchemy import create_engine, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Integer,String
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
import uuid
from datetime import datetime, timedelta


engine = create_engine(r'sqlite:///database.db',echo=False)

Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()


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

# 기존 테이블 삭제 / 생성
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

# READ
result = session.query(timeTable).all()
for row in result:
    print(f"\n시작시간: {row.initial_time} | 종료 시간: {row.end_time} | 날짜 : {row.date} \
    \n 총 시간 : {row.total_time} | 공부 시간 : {row.studying_time} | 딴짓 시간 : {row.playing_time}")




# db 전처리
# searchs = session.query(timeTable).filter(
#     timeTable.end_time.like('2023-10-09 21%')
# ).all()

# for search in searchs:
#     search.end_time = "2023-10-09 21"

# session.commit()    

# db groupby
# result = session.query(
#     func.sum(timeTable.total_time).label('total_time_sum'),
#     func.sum(timeTable.studying_time).label('studying_time_sum'),
#     func.sum(timeTable.playing_time).label('playing_time_sum'),
#     timeTable.end_time.label('end_time')
# ).group_by(timeTable.end_time).all()


# total_time,studying_time,playing_time,end_time = [],[],[],[]

# for row in result:
#     total_time.append(row[0])
#     studying_time.append(row[1])
#     playing_time.append(row[2])
#     end_time.append(row[3])
    

# print(total_time,studying_time,playing_time,end_time)

# 한시간 단위로 하고 나머지는 누적 



