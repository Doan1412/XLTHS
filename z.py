import torch

# Example list of tensors with Long dtype
tensor_list = [torch.tensor([1, 2, 3, 4], dtype=torch.long), torch.tensor([
    4, 3, 2, 1], dtype=torch.long)]

# Convert tensors to floating-point dtype
tensor_list_float = [tensor.float() for tensor in tensor_list]

# Stack tensors along a new dimension (dimension 0 in this case)
stacked_tensors = torch.stack(tensor_list_float, dim=0)

# Calculate the mean along the stacked dimension (dimension 0)
average_tensor = torch.mean(stacked_tensors, dim=0)

print("List of Tensors:")
for tensor in tensor_list:
    print(tensor)

print("\nAverage Tensor:")
print(average_tensor)
