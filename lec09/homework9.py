import numpy as np



def VAD(waveform, Fs):

    frame_length = int(0.025 * Fs)
    step = int(0.01 * Fs)

    energies = []
    starts = []

    for start in range(0, len(waveform) - frame_length, step):
        frame = waveform[start:start + frame_length]
        energies.append(np.sum(frame ** 2))
        starts.append(start)

    energies = np.array(energies)

    threshold = 0.1 * np.max(energies)

    segments = []
    current_segment = []

    for i, energy in enumerate(energies):

        if energy > threshold:

            start = starts[i]
            frame = waveform[start:start + frame_length]

            if len(current_segment) == 0:
                current_segment = list(frame)
            else:
                current_segment.extend(frame)

        else:

            if len(current_segment) > 0:
                segments.append(np.array(current_segment))
                current_segment = []

    if len(current_segment) > 0:
        segments.append(np.array(current_segment))

    return segments
   

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    ...
    '''

    N = int(0.004 * Fs)
    S = int(0.002 * Fs)

    models = []

    for segment in segments:

        frames = np.array([
            segment[m+1:m+1+N] - segment[m:m+N]
            for m in range(0, len(segment)-N, S)
        ])

        mstft = np.abs(np.fft.fft(frames, axis=1))

        sgram = np.log(1e-7 + mstft)

        model = np.average(
            sgram[:, :int(N/2)],
            axis=0
        )

        models.append(model)

    return models
def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.

    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model

    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''

    testsegments = VAD(testspeech, Fs)

    testmodels = segments_to_models(testsegments, Fs)

    sims = np.zeros((len(models), len(testmodels)))

    for y in range(len(models)):

        for k in range(len(testmodels)):

            x = testmodels[k]
            m = models[y]

            sims[y, k] = (
                np.dot(x, m)
                /
                (
                    np.sqrt(np.sum(x**2))
                    *
                    np.sqrt(np.sum(m**2))
                )
            )

    test_outputs = [
        labels[np.argmax(sims[:, k])]
        for k in range(len(testmodels))
    ]

    return sims, test_outputs