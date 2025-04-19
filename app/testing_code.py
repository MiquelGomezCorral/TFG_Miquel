
import time
import random as ran
from TFG.utils.time_traker import TimeTracker, parse_seconds_to_minutes
import time
from datetime import datetime

    
class Test():
    def __init__(self):
        self.test = "test"





if __name__ == "__main__":
    # N = 100
    # # TIME_TRACKER = TimeTracker(name="Testing", start_track_now=True)
    # TIME_TRACKER = TimeTracker(name="Testing")
    # time.sleep(0.1)
    # TIME_TRACKER.track("START", verbose=False)
    # for i in range(N):
    #     TIME_TRACKER.start_lap()
    #     time.sleep(0.5 + ran.random()*.5 - .25) 
    #     TIME_TRACKER.track("Load data")
    #     time.sleep(0.5 + ran.random()*.5 - .25)
        
    #     TIME_TRACKER.track("Parse model")
        
    #     TIME_TRACKER.track("Train model")
    #     time.sleep(1.5 + ran.random() - .5)
        
    #     TIME_TRACKER.track("Test model")
    #     time.sleep(0.5 + ran.random()*.5 - .25)
        
    #     TIME_TRACKER.finish_lap()
        
    # TIME_TRACKER.print_metrics(N)


    # t = 1744252067.5264242
    # readable = datetime.fromtimestamp(t)
    # print(readable)
    # print(parse_seconds_to_minutes(time.time() - t), "ago")
    
    # page_break = "\n\n ------- PAGE BREAK ------- \n\n"
    # raw_lines = page_break.join([
    #     "\n".join([line for line in page]) 
    #     for page in result.pages
    # ])
    
    # print(raw_lines)
    test: Test = Test()
    print(test.test)
    print(test.test2)
    
