from preprocess import clean_hindi_text

sample = "वह वहां गया। वे लोग वहां थे। यह सही है।"
print("Before:", sample)
print("After :", clean_hindi_text(sample))