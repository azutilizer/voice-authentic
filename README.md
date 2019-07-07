# Voice Authentication based on Live Streaming



## How to train

First, download ffmpeg for windows and put it under project directory.

	python3 train_DNN.py
	
Or you can train the model using UI.



## How to test

- Write <ID> and cleck Enter.
- If inputted ID is present voice, you can record your voice for 30 seconds and verify it.

  or if it is absent one, you'll be guided to record for 60 seconds and it'll be splitted, kept in database as voice sample.

  After doing it, you can verify your voice as above.
  
