# original code https://gist.github.com/Denbergvanthijs/7f6936ca90a683d37216fd80f5750e9c#file-donut-py

import time
import sys
import numpy as np
import threading

from TFG.scripts_dataset.utils import print_separator

stop_event = threading.Event()


screen_size = 20
view_box = screen_size + 5
theta_spacing = 0.07
phi_spacing = 0.02
illumination = np.fromiter(".,-~:;=!*#$@", dtype="<U1")
clear_bash_line = "\033[1A\x1b[2K"

A = 1
B = 1
R1 = 1
R2 = 2
K2 = 5
K1 = screen_size * K2 * 3 / (8 * (R1 + R2))


def render_frame(A: float, B: float) -> np.ndarray:
    """
    Returns a frame of the spinning 3D donut.
    Based on the pseudocode from: https://www.a1k0n.net/2011/07/20/donut-math.html
    """
    cos_A = np.cos(A)
    sin_A = np.sin(A)
    cos_B = np.cos(B)
    sin_B = np.sin(B)

    output = np.full((screen_size, screen_size), " ")  # (40, 40)
    zbuffer = np.zeros((screen_size, screen_size))  # (40, 40)

    cos_phi = np.cos(phi := np.arange(0, 2 * np.pi, phi_spacing))  # (315,)
    sin_phi = np.sin(phi)  # (315,)
    cos_theta = np.cos(theta := np.arange(0, 2 * np.pi, theta_spacing))  # (90,)
    sin_theta = np.sin(theta)  # (90,)
    circle_x = R2 + R1 * cos_theta  # (90,)
    circle_y = R1 * sin_theta  # (90,)

    x = (np.outer(cos_B * cos_phi + sin_A * sin_B * sin_phi, circle_x) - circle_y * cos_A * sin_B).T  # (90, 315)
    y = (np.outer(sin_B * cos_phi - sin_A * cos_B * sin_phi, circle_x) + circle_y * cos_A * cos_B).T  # (90, 315)
    z = ((K2 + cos_A * np.outer(sin_phi, circle_x)) + circle_y * sin_A).T  # (90, 315)
    ooz = np.reciprocal(z)  # Calculates 1/z
    xp = (screen_size / 2 + K1 * ooz * x).astype(int)  # (90, 315)
    yp = (screen_size / 2 - K1 * ooz * y).astype(int)  # (90, 315)
    L1 = (((np.outer(cos_phi, cos_theta) * sin_B) - cos_A * np.outer(sin_phi, cos_theta)) - sin_A * sin_theta)  # (315, 90)
    L2 = cos_B * (cos_A * sin_theta - np.outer(sin_phi, cos_theta * sin_A))  # (315, 90)
    L = np.around(((L1 + L2) * 8)).astype(int).T  # (90, 315)
    mask_L = L >= 0  # (90, 315)
    chars = illumination[L]  # (90, 315)

    for i in range(90):
        mask = mask_L[i] & (ooz[i] > zbuffer[xp[i], yp[i]])  # (315,)

        zbuffer[xp[i], yp[i]] = np.where(mask, ooz[i], zbuffer[xp[i], yp[i]])
        output[xp[i], yp[i]] = np.where(mask, chars[i], output[xp[i], yp[i]])

    return output


def pprint(array: np.ndarray) -> None:
    """Pretty print the frame."""
    print(*[" ".join(row) for row in array], sep="\n")

def print_donut(n_iters: int = screen_size * screen_size, infinite: bool = False, frame_delay: float = 0.01):
    print_separator("🍩 WELLCOME TO DONUT!    We are loading everything for you :) 🍩", sep_type="SUPER")    
    print("\n"*(screen_size))
    
    global A, B
    
    if infinite:
        n_iters = int(1e10)
        
    start_time = time.time() 
    for frame in range(n_iters):
    # while True:
        if stop_event.is_set():  # Check if the event is set
            print(*[clear_bash_line] * (view_box), sep="")
            break
        A += theta_spacing
        B += phi_spacing
        
        output = render_frame(A, B)
        # Delete as much lines as printed
        print(*[clear_bash_line] * (len(output)+1), sep="")
        pprint(output)
        
        # time.sleep(0.01)
        # Compute time taken and sleep only for remaining time
        elapsed_time = time.time() - start_time
        next_frame_time = (frame + 1) * frame_delay
        sleep_time = max(0, next_frame_time - elapsed_time)
        time.sleep(sleep_time)
            


if __name__ == "__main__":    
    print_donut(screen_size * screen_size)