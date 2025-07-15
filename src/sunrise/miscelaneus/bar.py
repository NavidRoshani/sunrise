import time
import sys

def giussepe_bar(step:int,total_steps:int,toolbar_width:int = 50):
    toolbar_width = total_steps*(toolbar_width//total_steps)
    time.sleep(0.1)
    i = step*(toolbar_width//total_steps)
    sys.stdout.write(f"\r\b[{(i-1)*'~'}><(((º>{(40-i-1)*' '}]")
    sys.stdout.flush()