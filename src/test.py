from jiwer import wer

ground_truth = "péa ha'e"
hypothesis =  "pé  ahaʼe"

error = wer(ground_truth, hypothesis)
print(error)
