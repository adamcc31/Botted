try:
    abs(None)
except TypeError as e:
    print(f"abs(None) error: {e}")

try:
    1.0 - None
except TypeError as e:
    print(f"1.0 - None error: {e}")
