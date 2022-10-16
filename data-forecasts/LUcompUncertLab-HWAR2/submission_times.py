#mcandrew

#--find next monday
def next_monday(dat=0, from_date=-1):
    from datetime import datetime,timedelta

    if from_date==-1:
        dt  = datetime.now()
    else:
        dt = datetime.strptime(from_date,"%Y-%m-%d")
    day = dt.weekday()

    counter = 0
    while day%7 !=0:
        counter+=1
        day+=1

    if dat:
        return (dt+timedelta(days=counter)).strftime("%Y-%m-%d")
    return counter

def next_saturday_after_monday_submission( num_days_until_monday, from_date=-1 ):
    from datetime import datetime, timedelta

    if from_date==-1:
        dt  = datetime.now()+ timedelta(days=num_days_until_monday)
    else:
        dt = datetime.strptime(from_date,"%Y-%m-%d")
        dt = dt + timedelta(days=num_days_until_monday)
        
    while dt.weekday() !=5 :
        dt = dt + timedelta(days=1)
    return dt.strftime("%Y-%m-%d")

def collect_target_end_dates(first_saturday):
    from datetime import datetime, timedelta
    sat = datetime.strptime(first_saturday,'%Y-%m-%d')

    target_end_dates = [first_saturday]
    for _ in range(3):
        sat =sat + timedelta(days=7)
        target_end_dates.append( sat.strftime("%Y-%m-%d") )
    return target_end_dates 

def define_submission_date(num_days_until_monday):
    from datetime import datetime, timedelta
    now  = datetime.now()
    monday_submission = now+timedelta(days=num_days_until_monday)

    return monday_submission.strftime("%Y-%m-%d")

def collect_model_name():
    import os
    return os.getcwd().split("/")[-1]

if __name__ == "__main__":
    pass

