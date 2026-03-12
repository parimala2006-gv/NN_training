from nltk.tokenize import WhitespaceTokenizer
text = "i love macnine learning and deep learning"
tokenizer = WhitespaceTokenizer()
tokens = tokenizer.tokenize(text)
print(tokens)
