import sys

alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
random_state = int(sys.argv[2]) if len(sys.argv) > 2 else 42

print("Alpha:", alpha)
print("Random State:", random_state)
