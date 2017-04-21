s = 'a b c d e'.split()
k = (3-1)//2

for i in range(len(s)):
  buf = ''
  if i-k < 0:
    for j in range(k):
      buf+='0'
  else:
    buf+=s[i-k]
  buf+=s[i]
  if i+k>=len(s):
    for j in range(k):
      buf+='0'
  else:
    buf+=s[i+k]
  print(buf)
