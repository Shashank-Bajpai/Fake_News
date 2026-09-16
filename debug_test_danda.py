import preprocess

# 1. Confirm which file Python is actually importing
print("Importing preprocess from:", preprocess.__file__)

# 2. Print the actual source code of clean_hindi_text as Python sees it
import inspect
print("\n--- Actual function source ---")
print(inspect.getsource(preprocess.clean_hindi_text))

# 3. Check the danda character itself
danda_in_sample = "गया।"[-1]
print("\nDanda char code point:", hex(ord(danda_in_sample)))