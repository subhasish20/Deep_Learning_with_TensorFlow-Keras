import tensorflow as tf

# Create a tensor from a list of values
tensor1 = tf.Tensor([1, 2, 3, 4, 5], dtype=tf.int32)

# Create a 2D tensor
tensor2 = tf.Tensor([[1, 2], [3, 4]], dtype=tf.float32)

# Print the tensors
print("Tensor 1:", tensor1)
print("Tensor 2:", tensor2)

# Check the type and shape
print("Type of Tensor 1:", type(tensor1))
print("Shape of Tensor 1:", tensor1.shape)
print("Dtype of Tensor 1:", tensor1.dtype)
