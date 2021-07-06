text=open("pall.txt","r").read()
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')