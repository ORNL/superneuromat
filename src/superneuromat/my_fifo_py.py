import os 


fifo_path = "./my_fifo"

if not os.path.exists(fifo_path):
	os.mkfifo(fifo_path)

data = 5;

with open(fifo_path, "wb") as fifo:
	fifo.write(data)

