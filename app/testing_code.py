
import time
import random as ran
from TFG.scripts_dataset.time_traker import TimeTracker


if __name__ == "__main__":
    N = 100
    # TIME_TRACKER = TimeTracker(name="Testing", start_track_now=True)
    TIME_TRACKER = TimeTracker(name="Testing")
    time.sleep(0.1)
    TIME_TRACKER.track("START", verbose=False)
    for i in range(N):
        TIME_TRACKER.start_lap()
        time.sleep(0.5 + ran.random()*.5 - .25) 
        TIME_TRACKER.track("Load data")
        time.sleep(0.5 + ran.random()*.5 - .25)
        
        TIME_TRACKER.track("Parse model")
        
        TIME_TRACKER.track("Train model")
        time.sleep(1.5 + ran.random() - .5)
        
        TIME_TRACKER.track("Test model")
        time.sleep(0.5 + ran.random()*.5 - .25)
        
        TIME_TRACKER.finish_lap()
        
    TIME_TRACKER.print_metrics(N)