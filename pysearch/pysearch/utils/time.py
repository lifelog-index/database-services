
from datetime import datetime
import parsedatetime

def time_this(func):
    def calc_time(*args, **kwargs):
        before = datetime.now()
        x = func(*args, **kwargs)
        after = datetime.now()
        print("Function {} elapsed time: {}".format(func.__name__, after-before))
        return x
    return calc_time


def nlp2datetime(text: str):
    cal = parsedatetime.Calendar()
    time_struct, parse_status = cal.parse(text)#, sourceTime=start)
    if parse_status == 0:
        return None
    return datetime(*time_struct[:6])

def hour2part_of_day(hour: int):
    if (hour >= 5) and (hour < 9):
        return 'early morning'
    elif (hour >= 9) and (hour < 11):
        return 'morning'
    elif (hour >= 11) and (hour < 12):
        return 'late morning'
    elif (hour >= 12) and (hour < 15):
        return 'early afternoon'
    elif (hour >= 15) and (hour < 16):
        return 'afternoon'
    elif (hour >= 16) and (hour < 17):
        return 'late afternoon'
    elif (hour >= 17) and (hour < 19):
        return 'early evening'
    elif (hour >= 19) and (hour < 21):
        return 'early evening'
    elif (hour >= 21) and (hour < 23):
        return 'late afternoon'
    else:
        return 'night'
        
def day2day_of_week(day: int):
    day_of_week = [
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday',
    ]
    return day_of_week[day]

def month2month_of_year(month: int):
    month_of_year = [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December',
    ]
    assert month <= 12, f'Invalid month: {month}'
    assert month > 0, f'Invalid month: {month}'
    return month_of_year[month-1]