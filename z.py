# Example dictionary
my_dict = {'key1': 5, 'key2': 3, 'key3': 8, 'key4': 1}

# Find the key with the minimum value
min_key = min(my_dict, key=lambda k: my_dict[k])

print("Key with minimum value:", min_key)
