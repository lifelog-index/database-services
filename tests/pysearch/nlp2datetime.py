from datetime import datetime
import time
from pysearch.utils.time import nlp2datetime

now_parsed = nlp2datetime("1 month before now")
now = datetime(time.localtime().tm_year, time.localtime().tm_mon, time.localtime().tm_mday, time.localtime().tm_hour, time.localtime().tm_min, time.localtime().tm_sec)
print('now: ', now)
print('parse time: ', now_parsed)