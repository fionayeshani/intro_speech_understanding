import numpy as np
import librosa
from scipy.signal import lfilter

def lpc(speech, frame_length, frame_skip, order):
    
    nframes = int((len(speech)-frame_length)/frame_skip)
    frames = np.array([speech[m*frame_skip:m*frame_skip+frame_length] for m in range(nframes)])
    A = librosa.lpc(frames, order=order)
    excitation = np.zeros((nframes, frame_length))
    for m in range(nframes):
        for n in range(order, frame_length):
            for k in range(0, order+1):
                excitation[m,n] += A[m,k]*frames[m,n-k]
    return A, excitation

def synthesize(e, A, frame_skip):

    nframes = A.shape[0]
    synthesis = np.zeros(len(e))

    for m in range(nframes):

        start = m * frame_skip
        end = start + frame_skip

        synthesis[start:end] = lfilter([1], A[m], e[start:end])

    return synthesis


def robot_voice(excitation, T0, frame_skip):
    nframes, frame_length = excitation.shape

    gain = np.zeros(nframes)

    for m in range(nframes):
        gain[m] = np.sqrt(np.average(np.square(excitation[m, :])))

    e_robot = np.zeros(nframes * frame_skip)

    n = 0
    while n < len(e_robot):
        e_robot[n] = gain[int(n / frame_skip)]
        n += T0

    return gain, e_robot