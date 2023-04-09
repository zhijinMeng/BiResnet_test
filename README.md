This file records how to use this test code for BiResnet Model that built on SPR-Sound Dataset.
1. Put testdata_original folder in the working dir of 'SPR_BiResnet_split_test/'
2. when 'testdata_original' folder is uploaded, excute the following python files in order to convert the data into appropriate form for testing and run the test eventually:
    <br>
    python split.py
    <br>
    python stft.py
    <br>
    python wavelet.py
    <br>
    python pack.py
    <br>
    python test.py