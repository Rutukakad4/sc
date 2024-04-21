def complement(A):
    """Function to compute complement of a fuzzy set."""
    return [1 - x for x in A]

def intersection(A, B):
    """Function to compute intersection of two fuzzy sets."""
    return [min(A[i], B[i]) for i in range(len(A))]

def union(A, B):
    """Function to compute union of two fuzzy sets."""
    return [max(A[i], B[i]) for i in range(len(A))]

def demorgans_law1(A, B):
    """Implement DeMorgan's law 1: complement of the union equals intersection of complements."""
    complement_union = complement(union(A, B))
    intersection_complements = intersection(complement(A), complement(B))
    return complement_union, intersection_complements

def demorgans_law2(A, B):
    """Implement DeMorgan's law 2: complement of the intersection equals union of complements."""
    complement_intersection = complement(intersection(A, B))
    union_complements = union(complement(A), complement(B))
    return complement_intersection, union_complements

# Example fuzzy sets
A = [0.2, 0.4, 0.6, 0.8]
B = [0.3, 0.5, 0.7, 0.9]

# Applying DeMorgan's laws
complement_union, intersection_complements = demorgans_law1(A, B)
complement_intersection, union_complements = demorgans_law2(A, B)

# Displaying results
print("DeMorgan's Law 1:")
print("Complement of Union:", complement_union)
print("Intersection of Complements:", intersection_complements)

print("\nDeMorgan's Law 2:")
print("Complement of Intersection:", complement_intersection)
print("Union of Complements:", union_complements)
