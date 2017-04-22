import numpy as np

k = 1

s = [1,2,3,4,5]
s = np.pad(s, (k), 'constant', constant_values=(0))
for i in range(k, len(s)-k):
  buf = ''
  for j in range(i-k, i+k+1):
    buf += str(s[j])
  print(buf)

print('*' * 80)

def slide(i):
  buf = []
  for j in range(i-k, i+k+1):
    buf.append(s[j])
  return buf

print(list(map(slide, range(k, len(s)-k))))